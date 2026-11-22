"""
Торгівля на демо-рахунку OKX за сигналами бота.

Логіка така сама, як у віртуального супроводу:
  • вхід за ринком, стоп одразу ставиться на біржі (спрацює, навіть якщо бот у цей момент не працює);
  • ½ закривається там же, де віртуальна угода фіксує ½, після цього стоп переноситься в беззбиток;
  • решта закривається на виході / стопі / за часом.

Розмір позиції: RISK_PCT % від реального балансу USDT на демо-рахунку (береться з OKX перед кожною угодою) —
тобто компаунд. Якщо задано OKX_START_BALANCE — замість цього віртуальний баланс від цієї суми + реальний PnL.

Секрети: OKX_DEMO_KEY, OKX_DEMO_SECRET, OKX_DEMO_PASSPHRASE.
"""
import base64
import hashlib
import hmac
import json
import math
import os
import time
from datetime import datetime, timezone

import requests

BASE = "https://www.okx.com"


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


class OkxError(RuntimeError):
    pass


class OkxDemoTrader:
    def __init__(self, inst_map, notify, start_balance=None, risk_pct=None, leverage=None, session=None):
        """inst_map: {"BTCUSDT": "BTC-USDT-SWAP", ...}; notify(text, reply_to=None) — повідомлення в Telegram."""
        self.key = _env("OKX_DEMO_KEY")
        self.secret = _env("OKX_DEMO_SECRET")
        self.passphrase = _env("OKX_DEMO_PASSPHRASE")
        if not (self.key and self.secret and self.passphrase):
            raise OkxError("не задані секрети OKX_DEMO_KEY / OKX_DEMO_SECRET / OKX_DEMO_PASSPHRASE")
        self.s = session or requests.Session()
        self.notify = notify
        sb = start_balance if start_balance is not None else _env("OKX_START_BALANCE")
        self.start_balance = float(sb) if sb else None  # None — працюємо від реального балансу USDT
        self.risk_pct = float(risk_pct if risk_pct is not None else _env("RISK_PCT", "1"))
        self.leverage = int(leverage if leverage is not None else _env("OKX_LEVERAGE", "20"))
        self.inst_map = dict(inst_map)
        self.specs = {}
        self.skipped = {}
        self._setup()

    # ------------------------------------------------------------------ HTTP

    def _req(self, method, path, params=None, body=None, auth=True):
        if params:
            path = path + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        data = json.dumps(body) if body is not None else ""
        headers = {"Content-Type": "application/json", "x-simulated-trading": "1"}
        if auth:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
                f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
            sign = base64.b64encode(hmac.new(self.secret.encode(), (ts + method + path + data).encode(),
                                             hashlib.sha256).digest()).decode()
            headers.update({"OK-ACCESS-KEY": self.key, "OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": ts,
                            "OK-ACCESS-PASSPHRASE": self.passphrase})
        for attempt in range(4):
            r = self.s.request(method, BASE + path, data=data or None, headers=headers, timeout=20)
            if r.status_code == 429:
                time.sleep(1 + attempt)
                continue
            try:
                j = r.json()
            except Exception:
                raise OkxError(f"{method} {path}: HTTP {r.status_code} {r.text[:200]}")
            if str(j.get("code")) != "0":
                det = j.get("data") or []
                sub = "; ".join(f"{d.get('sCode')} {d.get('sMsg')}" for d in det if isinstance(d, dict) and d.get("sMsg"))
                raise OkxError(f"{path}: {j.get('code')} {j.get('msg')} {sub}".strip())
            return j.get("data", [])
        raise OkxError(f"{path}: забагато запитів (429)")

    # ------------------------------------------------------------------ налаштування

    def _setup(self):
        cfg = self._req("GET", "/api/v5/account/config")[0]
        if str(cfg.get("acctLv")) == "1":  # «Спот» — ф'ючерси недоступні; переводимо в «Спот і ф'ючерси»
            self._req("POST", "/api/v5/account/set-account-level", body={"acctLv": "2"})
        if cfg.get("posMode") != "long_short_mode":  # щоб LONG і SHORT по одному активу не гасили одне одного
            try:
                self._req("POST", "/api/v5/account/set-position-mode", body={"posMode": "long_short_mode"})
            except OkxError as e:
                print(f"OKX: режим позицій не змінено: {e}")
        for sym, inst in list(self.inst_map.items()):
            try:
                d = self._req("GET", "/api/v5/public/instruments", {"instType": "SWAP", "instId": inst}, auth=False)
                if not d or d[0].get("state") != "live":
                    raise OkxError("інструмента немає на демо-рахунку")
                spec = d[0]
                if spec.get("ctType") != "linear":
                    raise OkxError("не лінійний контракт")
                self.specs[sym] = {k: float(spec[k]) for k in ("ctVal", "lotSz", "minSz", "tickSz")}
                self.specs[sym]["inst"] = inst
                lev = min(self.leverage, int(float(spec.get("lever") or self.leverage)))
                try:
                    self._req("POST", "/api/v5/account/set-leverage",
                              body={"instId": inst, "lever": str(lev), "mgnMode": "cross"})
                except OkxError as e:
                    print(f"OKX {inst}: плече не змінено: {e}")
            except Exception as e:
                self.skipped[sym] = str(e)
                self.specs.pop(sym, None)

    def can_trade(self, sym):
        return sym in self.specs

    def status_lines(self):
        out = [f"• {s} → {v['inst']}" for s, v in self.specs.items()]
        out += [f"• {s}: лише сигнали ({why})" for s, why in self.skipped.items()]
        return out

    # ------------------------------------------------------------------ допоміжне

    @staticmethod
    def _round_down(x, step):
        return math.floor(x / step + 1e-9) * step

    @staticmethod
    def _fmt_num(x, step):
        dec = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        return f"{x:.{dec}f}"

    def _px(self, sym, px):
        sp = self.specs[sym]
        return self._fmt_num(round(px / sp["tickSz"]) * sp["tickSz"], sp["tickSz"])

    def _sz(self, sym, sz):
        return self._fmt_num(sz, self.specs[sym]["lotSz"])

    def account_usdt(self):
        d = self._req("GET", "/api/v5/account/balance", {"ccy": "USDT"})
        for x in (d[0].get("details") or []) if d else []:
            if x.get("ccy") == "USDT":
                return float(x.get("eq") or x.get("cashBal") or 0)
        return 0.0

    def equity(self, state):
        """Баланс, від якого рахується ризик. Перший виклик запам'ятовує стартову суму (okx_start)."""
        if self.start_balance is not None:
            eq = float(state.get("okx_equity", self.start_balance))
            state.setdefault("okx_start", self.start_balance)
            return eq
        try:
            eq = self.account_usdt()
            state["okx_equity"] = round(eq, 4)
        except Exception as e:
            print(f"OKX баланс: {e}")
            eq = float(state.get("okx_equity", 0))
        state.setdefault("okx_start", round(eq, 2))
        return eq

    def _order_fill(self, inst, ord_id):
        for _ in range(6):
            d = self._req("GET", "/api/v5/trade/order", {"instId": inst, "ordId": ord_id})
            if d and d[0].get("state") == "filled":
                return float(d[0]["avgPx"]), float(d[0]["accFillSz"])
            time.sleep(0.5)
        raise OkxError("ордер не виконався")

    def _mark(self, sym):
        d = self._req("GET", "/api/v5/public/mark-price", {"instType": "SWAP", "instId": self.specs[sym]["inst"]},
                      auth=False)
        return float(d[0]["markPx"])

    def _order_info(self, inst, ord_id):
        """(середня ціна, виконано контрактів, комісія ≤ 0) по ордеру."""
        for _ in range(8):
            d = self._req("GET", "/api/v5/trade/order", {"instId": inst, "ordId": ord_id})
            if d and d[0].get("state") == "filled":
                return float(d[0]["avgPx"]), float(d[0]["accFillSz"]), float(d[0].get("fee") or 0)
            time.sleep(0.5)
        raise OkxError("ордер не виконався")

    def _market(self, sym, side, opening, sz):
        """Ринковий ордер на sz контрактів; opening=False — закриття частини позиції side."""
        sp = self.specs[sym]
        buy = (side == "LONG") == opening
        d = self._req("POST", "/api/v5/trade/order", body={
            "instId": sp["inst"], "tdMode": "cross", "side": "buy" if buy else "sell",
            "posSide": side.lower(), "ordType": "market", "sz": self._sz(sym, sz)})
        return self._order_info(sp["inst"], d[0]["ordId"])

    def _place_sl(self, sym, side, stop, sz):
        """Стоп саме на обсяг цієї угоди (а не на всю позицію) — щоб кілька угод в один бік жили незалежно.
        Тригер — mark price: вона йде за реальним ринком, а не за тонким стаканом демо-рахунку."""
        sp = self.specs[sym]
        d = self._req("POST", "/api/v5/trade/order-algo", body={
            "instId": sp["inst"], "tdMode": "cross", "side": "sell" if side == "LONG" else "buy",
            "posSide": side.lower(), "ordType": "conditional", "sz": self._sz(sym, sz),
            "slTriggerPx": self._px(sym, stop), "slOrdPx": "-1", "slTriggerPxType": "mark"})
        return d[0]["algoId"]

    def _cancel_sl(self, sym, algo_id):
        if not algo_id:
            return
        try:
            self._req("POST", "/api/v5/trade/cancel-algos", body=[{"algoId": algo_id, "instId": self.specs[sym]["inst"]}])
        except OkxError as e:
            print(f"OKX: скасування стопу {algo_id}: {e}")

    def _position_sz(self, sym, side):
        d = self._req("GET", "/api/v5/account/positions", {"instType": "SWAP", "instId": self.specs[sym]["inst"]})
        return sum(abs(float(p.get("pos") or 0)) for p in d if p.get("posSide") == side.lower())

    def _book(self, sym, info, side, px, done, fee):
        """Записати закриття частини угоди: PnL цієї частини від ціни входу цієї угоди + комісія."""
        sg = 1 if side == "LONG" else -1
        info["pnl_acc"] = info.get("pnl_acc", 0.0) + sg * (px - info["avg"]) * done * self.specs[sym]["ctVal"]
        info["fee_acc"] = info.get("fee_acc", 0.0) + fee
        info["left"] = max(0.0, round(info["left"] - done, 8))

    # ------------------------------------------------------------------ події угоди

    def on_open(self, state, t):
        """Відкрити реальну позицію під віртуальну угоду t (з bot.open_trade). Кожна угода — окремий обсяг і окремий стоп,
        навіть якщо в цей бік уже є позиція (добір)."""
        sym, side = t["sym"], t["side"]
        if not self.can_trade(sym):
            return
        sp = self.specs[sym]
        eq = self.equity(state)
        risk_usd = eq * self.risk_pct / 100
        if risk_usd <= 0:
            self.notify("💼 Угоду не відкрито: на демо-рахунку немає USDT", t.get("mid"))
            return
        try:  # розмір — від поточної ціни, а не від ціни сигналу: вхід міг статися пізніше
            ref = self._mark(sym)
        except Exception as e:
            print(f"OKX mark {sp['inst']}: {e}")
            ref = t["entry"]
        if (side == "LONG" and ref <= t["stop"]) or (side == "SHORT" and ref >= t["stop"]):
            self.notify(f"💼 Угоду не відкрито: ціна {ref:g} вже за стопом {self._px(sym, t['stop'])}", t.get("mid"))
            return
        dist = abs(ref - t["stop"])
        sz = self._round_down(risk_usd / (dist * sp["ctVal"]), sp["lotSz"])
        if sz < sp["minSz"]:
            self.notify(f"💼 Угоду не відкрито: ризик ${risk_usd:.2f} замалий для мінімального лота {sp['inst']}",
                        t.get("mid"))
            return
        cut = ""
        try:  # скільки контрактів дозволяє маржа; якщо 1% не влазить — відкриваємо менше
            m = self._req("GET", "/api/v5/account/max-size", {"instId": sp["inst"], "tdMode": "cross"})
            mx = float(m[0]["maxBuy" if side == "LONG" else "maxSell"]) if m else 0
            mx = self._round_down(mx * 0.9, sp["lotSz"])
            if sz > mx:
                if mx < sp["minSz"]:
                    self.notify(f"💼 Угоду не відкрито: не вистачає маржі на мінімальний лот {sp['inst']}", t.get("mid"))
                    return
                cut = f"\n⚠️ Ризик урізано маржею (стоп {dist / ref * 100:.3f}% — 1% не влазить у плече)"
                sz = mx
        except Exception as e:
            print(f"OKX max-size {sp['inst']}: {e}")
        adds = [x for x in state.get("trades", []) if x is not t and x["sym"] == sym and x["side"] == side
                and x.get("okx") and not x["okx"].get("closed")]
        try:
            avg, filled, fee = self._market(sym, side, True, sz)
        except Exception as e:
            self.notify(f"💼 ⚠️ Помилка входу {sp['inst']}: {e}", t.get("mid"))
            return
        real_risk = abs(avg - t["stop"]) * filled * sp["ctVal"]
        info = dict(inst=sp["inst"], sz=filled, left=filled, avg=avg, ts=int(time.time() * 1000) - 5000,
                    risk=real_risk, eq=eq, algo=None, pnl_acc=0.0, fee_acc=fee, v=2)
        t["okx"] = info
        try:
            info["algo"] = self._place_sl(sym, side, t["stop"], filled)
            sl_txt = f"стоп {self._px(sym, t['stop'])} на {self._sz(sym, filled)} контр."
        except Exception as e:
            self.notify(f"💼 {sp['inst']} {side}: ⚠️ стоп НЕ поставлено ({e}) — закриваю цю угоду", t.get("mid"))
            self.on_close(state, t, "стоп не встановлено")
            return
        add_txt = (f"\n➕ Добір: у позиції {side} тепер {len(adds) + 1} угоди, у кожної свій стоп і вихід"
                   if adds else "")
        self.notify(f"💼 Відкрито {side} {self._sz(sym, filled)} контр. {sp['inst']} @{avg:g} "
                    f"(≈${filled * sp['ctVal'] * avg:,.0f}) · ризик ${real_risk:.2f} = {real_risk / eq * 100:.2f}% "
                    f"від ${eq:,.2f} · {sl_txt}{cut}{add_txt}", t.get("mid"))

    def on_half(self, state, t):
        """Зафіксувати ½ ОБСЯГУ ЦІЄЇ угоди і перенести її стоп у беззбиток (ціна її входу)."""
        info = t.get("okx")
        if not info or info.get("closed"):
            return
        sym, side = t["sym"], t["side"]
        sp = self.specs.get(sym)
        if not sp:
            return
        half = self._round_down(info["left"] / 2, sp["lotSz"])
        msg = []
        if half >= sp["minSz"]:
            try:
                px, done, fee = self._market(sym, side, False, half)
                self._book(sym, info, side, px, done, fee)
                msg.append(f"закрито ½ угоди ({self._sz(sym, done)} контр.) @{px:g}")
            except Exception as e:
                msg.append(f"⚠️ ½ не закрито: {e}")
        else:
            msg.append("½ не ділиться (мінімальний лот) — тримаю весь обсяг угоди")
        try:
            self._cancel_sl(sym, info.get("algo"))
            info["algo"] = None
            if info["left"] > 0:
                info["algo"] = self._place_sl(sym, side, info["avg"], info["left"])
                msg.append(f"стоп у БУ {self._px(sym, info['avg'])} на {self._sz(sym, info['left'])} контр.")
        except Exception as e:
            msg.append(f"⚠️ стоп у БУ не перенесено: {e}")
        self.notify("💼 " + " · ".join(msg), t.get("mid"))

    def check_stop(self, state, t):
        """Чи спрацював на біржі стоп цієї угоди. Якщо так — записує результат і повертає ціну виконання."""
        info = t.get("okx")
        if not info or info.get("closed") or not info.get("algo") or info.get("v") != 2:
            return None
        sym, side = t["sym"], t["side"]
        if sym not in self.specs:
            return None
        d = self._req("GET", "/api/v5/trade/order-algo", {"algoId": info["algo"]})
        if not d or d[0].get("state") != "effective":
            return None
        ord_id = d[0].get("ordId") or (d[0].get("ordIdList") or [None])[0]
        px = float(d[0].get("actualPx") or 0) or None
        try:
            if ord_id:
                px, done, fee = self._order_info(info["inst"], ord_id)
                self._book(sym, info, side, px, done, fee)
        except Exception as e:
            print(f"OKX стоп-ордер {ord_id}: {e}")
        info["algo"] = None
        info["stopped"] = True
        return px if px is not None else t["stop"]

    def on_close(self, state, t, how=""):
        """Закрити залишок САМЕ ЦІЄЇ угоди (інші угоди в позиції не чіпаємо), порахувати її PnL, оновити баланс."""
        info = t.get("okx")
        if not info or info.get("closed"):
            return
        sym, side = t["sym"], t["side"]
        sp = self.specs.get(sym)
        inst = info["inst"]
        info["closed"] = True
        err = ""
        if sp:
            self._cancel_sl(sym, info.get("algo"))
            info["algo"] = None
        if info.get("v") != 2:  # угоди, відкриті старою версією (стоп на всю позицію)
            try:
                if sp and self._position_sz(sym, side) > 0 and info.get("left", 0) > 0:
                    px, done, fee = self._market(sym, side, False, min(info["left"], self._position_sz(sym, side)))
            except Exception as e:
                err = f" · ⚠️ закриття: {e}"
            time.sleep(1.5)
            pnl, fee = self._realized(inst, side, info["ts"])
        else:
            try:
                if sp and info["left"] > 0 and not info.get("stopped"):
                    pos = self._position_sz(sym, side)
                    q = self._round_down(min(info["left"], pos), sp["lotSz"])
                    if q > 0:
                        px, done, fee_c = self._market(sym, side, False, q)
                        self._book(sym, info, side, px, done, fee_c)
            except Exception as e:
                err = f" · ⚠️ закриття: {e}"
            fee = info.get("fee_acc", 0.0)
            pnl = info.get("pnl_acc", 0.0) + fee
        if self.start_balance is not None:
            eq1 = self.equity(state) + pnl
            state["okx_equity"] = round(eq1, 4)
        else:
            eq1 = self.equity(state)
        info.update(pnl=round(pnl, 4), fee=round(fee, 4))
        r_real = pnl / info["risk"] if info.get("risk") else 0
        self.notify(f"💼 Закрито {inst} {side} ({how}): {'+' if pnl >= 0 else '−'}${abs(pnl):.2f} "
                    f"({r_real:+.2f}R реально, комісії ${abs(fee):.2f}) · баланс ${eq1:,.2f}{err}", t.get("mid"))

    def migrate(self, state):
        """Перевести угоди старої версії (стоп closeFraction на всю позицію) на окремі стопи по обсягу угоди."""
        out = []
        for t in state.get("trades", []):
            info = t.get("okx")
            if not info or info.get("closed") or info.get("v") == 2 or t["sym"] not in self.specs:
                continue
            sym, side = t["sym"], t["side"]
            tot, fee = self._realized(info["inst"], side, info["ts"])  # що вже зафіксовано (½) + комісії
            info["pnl_acc"], info["fee_acc"] = tot - fee, fee
            pos = self._position_sz(sym, side)
            info["left"] = min(info.get("left", 0), pos)
            self._cancel_sl(sym, info.get("algo"))
            info["algo"] = None
            info["v"] = 2
            if info["left"] <= 0:
                info["stopped"] = True  # позиції вже немає (стоп спрацював на біржі)
                out.append(f"{info['inst']} {side}: позиції вже немає на біржі")
                continue
            stop = info["avg"] if t.get("half") is not None else t["stop"]
            info["algo"] = self._place_sl(sym, side, stop, info["left"])
            out.append(f"{info['inst']} {side}: стоп {self._px(sym, stop)} на {self._sz(sym, info['left'])} контр.")
        return out

    def _realized(self, inst, side, since_ms):
        """Сума fillPnl + fee по угодах цієї позиції з моменту входу."""
        pnl = fee = 0.0
        try:
            fills = self._req("GET", "/api/v5/trade/fills-history",
                              {"instType": "SWAP", "instId": inst, "begin": str(since_ms), "limit": "100"})
        except Exception as e:
            print(f"OKX fills: {e}")
            return 0.0, 0.0
        for f in fills:
            if f.get("posSide") != side.lower():
                continue
            pnl += float(f.get("fillPnl") or 0)
            fee += float(f.get("fee") or 0)  # комісія від'ємна
        return pnl + fee, fee

    def reset_all(self):
        """Закрити всі SWAP-позиції і зняти всі умовні ордери (стопи) на демо-рахунку."""
        closed, cancelled = [], 0
        try:
            algos = self._req("GET", "/api/v5/trade/orders-algo-pending", {"instType": "SWAP", "ordType": "conditional"})
            if algos:
                self._req("POST", "/api/v5/trade/cancel-algos",
                          body=[{"algoId": a["algoId"], "instId": a["instId"]} for a in algos])
                cancelled = len(algos)
        except OkxError as e:
            print(f"OKX скасування стопів: {e}")
        for p in self._req("GET", "/api/v5/account/positions", {"instType": "SWAP"}):
            if abs(float(p.get("pos") or 0)) > 0:
                try:
                    self._req("POST", "/api/v5/trade/close-position", body={
                        "instId": p["instId"], "mgnMode": p.get("mgnMode") or "cross",
                        "posSide": p.get("posSide") or "net", "autoCxl": True})
                    closed.append(f"{p['instId']} {p.get('posSide')}")
                except OkxError as e:
                    closed.append(f"{p['instId']} ⚠️ {e}")
        return closed, cancelled

