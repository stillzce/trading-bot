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

    def _place_sl(self, sym, side, stop):
        sp = self.specs[sym]
        close_side = "sell" if side == "LONG" else "buy"
        d = self._req("POST", "/api/v5/trade/order-algo", body={
            "instId": sp["inst"], "tdMode": "cross", "side": close_side, "posSide": side.lower(),
            "ordType": "conditional", "slTriggerPx": self._px(sym, stop), "slOrdPx": "-1",
            "slTriggerPxType": "last", "closeFraction": "1", "reduceOnly": "true"})
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

    # ------------------------------------------------------------------ події угоди

    def on_open(self, state, t):
        """Відкрити реальну позицію під віртуальну угоду t (з bot.open_trade)."""
        sym, side = t["sym"], t["side"]
        if not self.can_trade(sym):
            return
        sp = self.specs[sym]
        eq = self.equity(state)
        risk_usd = eq * self.risk_pct / 100
        dist = abs(t["entry"] - t["stop"])
        if dist <= 0:
            return
        sz = self._round_down(risk_usd / (dist * sp["ctVal"]), sp["lotSz"])
        if risk_usd <= 0:
            self.notify(f"💼 Угоду не відкрито: на демо-рахунку немає USDT", t.get("mid"))
            return
        if sz < sp["minSz"]:
            self.notify(f"💼 Угоду не відкрито: ризик ${risk_usd:.2f} замалий для мінімального лота {sp['inst']} "
                        f"({self._sz(sym, sp['minSz'])} контр. = ${sp['minSz'] * sp['ctVal'] * dist:.2f} ризику)",
                        t.get("mid"))
            return
        try:
            d = self._req("POST", "/api/v5/trade/order", body={
                "instId": sp["inst"], "tdMode": "cross", "side": "buy" if side == "LONG" else "sell",
                "posSide": side.lower(), "ordType": "market", "sz": self._sz(sym, sz)})
            avg, filled = self._order_fill(sp["inst"], d[0]["ordId"])
        except Exception as e:
            self.notify(f"💼 ⚠️ Помилка входу {sp['inst']}: {e}", t.get("mid"))
            return
        ts = int(time.time() * 1000) - 5000
        info = dict(inst=sp["inst"], sz=filled, left=filled, avg=avg, ts=ts, risk=risk_usd, eq=eq, algo=None)
        t["okx"] = info
        try:
            info["algo"] = self._place_sl(sym, side, t["stop"])
            sl_txt = f"стоп на біржі {self._px(sym, t['stop'])}"
        except Exception as e:
            sl_txt = f"⚠️ стоп НЕ поставлено ({e}) — закриваю"
            self.notify(f"💼 {sp['inst']} {side}: {sl_txt}", t.get("mid"))
            self.on_close(state, t, "стоп не встановлено")
            return
        notional = filled * sp["ctVal"] * avg
        self.notify(f"💼 Відкрито {side} {self._sz(sym, filled)} контр. {sp['inst']} @{avg:g} "
                    f"(≈${notional:,.0f}) · ризик ${risk_usd:.2f} = {self.risk_pct:g}% від ${eq:,.2f} · {sl_txt}",
                    t.get("mid"))

    def on_half(self, state, t):
        """Зафіксувати ½ і перенести стоп у беззбиток."""
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
                d = self._req("POST", "/api/v5/trade/order", body={
                    "instId": sp["inst"], "tdMode": "cross", "side": "sell" if side == "LONG" else "buy",
                    "posSide": side.lower(), "ordType": "market", "sz": self._sz(sym, half)})
                px, done = self._order_fill(sp["inst"], d[0]["ordId"])
                info["left"] = max(0.0, info["left"] - done)
                msg.append(f"закрито ½ ({self._sz(sym, done)} контр.) @{px:g}")
            except Exception as e:
                msg.append(f"⚠️ ½ не закрито: {e}")
        else:
            msg.append("½ не ділиться (мінімальний лот) — тримаю всю позицію")
        try:
            self._cancel_sl(sym, info.get("algo"))
            info["algo"] = self._place_sl(sym, side, info["avg"])
            msg.append(f"стоп у БУ {self._px(sym, info['avg'])}")
        except Exception as e:
            msg.append(f"⚠️ стоп у БУ не перенесено: {e}")
        self.notify("💼 " + " · ".join(msg), t.get("mid"))

    def on_close(self, state, t, how=""):
        """Закрити залишок на біржі (якщо ще є), порахувати реальний PnL з комісіями, оновити баланс."""
        info = t.get("okx")
        if not info or info.get("closed"):
            return
        sym, side = t["sym"], t["side"]
        sp = self.specs.get(sym)
        inst = info["inst"]
        info["closed"] = True
        err = ""
        try:
            if sp and self._position_sz(sym, side) > 0:
                self._req("POST", "/api/v5/trade/close-position",
                          body={"instId": inst, "mgnMode": "cross", "posSide": side.lower(), "autoCxl": True})
        except Exception as e:
            err = f" · ⚠️ закриття: {e}"
        if sp:
            self._cancel_sl(sym, info.get("algo"))
        time.sleep(1.5)
        pnl, fee = self._realized(inst, side, info["ts"])
        if self.start_balance is not None:
            eq1 = self.equity(state) + pnl
            state["okx_equity"] = round(eq1, 4)
        else:
            eq1 = self.equity(state)
        info.update(pnl=round(pnl, 4), fee=round(fee, 4))
        r_real = pnl / info["risk"] if info.get("risk") else 0
        self.notify(f"💼 Закрито {inst} {side} ({how}): {'+' if pnl >= 0 else '−'}${abs(pnl):.2f} "
                    f"({r_real:+.2f}R реально, комісії ${abs(fee):.2f}) · баланс ${eq1:,.2f}{err}", t.get("mid"))

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
