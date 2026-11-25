"""Мультитаймфреймовий бектест (python bot.py mtf), крипта 24/7.

Модель (як торгує Богдан руками):
  1. Бот по колу дивиться всі ТФ. Дно WT (≤ −60) на будь-якому ТФ відкриває сетап у лонг (вершина ≥ 60 — у шорт).
     Дно = мінімум хвилі, якою WT заходив у зону. Тінь нижче — дно зсувається; тіло нижче — сетап зламано.
  2. Базовий ТФ — найстарший ТФ, на якому за час сетапу теж було дно WT.
  3. Вхід — закріп тіла за FVG проти нас. Кандидати: незаповнені FVG проти на базовому ТФ і всіх старших,
     що стоять над дном і не далі NEAR×ATR(M15) від ціни. Якщо є FVG старшого ТФ — валідний лише закріп за ним
     (за найстаршим). Якщо поруч нічого немає — FVG базового ТФ, що утворився під час сетапу.
     ENTRY_TF="fvg": закріп рахується закриттям свічки ТФ цього FVG; "base": закриттям свічки базового ТФ.
  4. Стоп — під дном. Вихід — протилежний сигнал за тими самими правилами (EXIT="any": будь-який;
     "base": лише з базовим ТФ не молодшим за базовий ТФ входу), або стоп, або 10 днів.
  Контекст дня (доба UTC): входимо лише в його бік (CTX=1) або в обидва боки (CTX=0). Вихід — незалежно від контексту.
  Комісія OKX 0.1% від обсягу (вхід + вихід) віднімається в R.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import bot
from bt4 import load, resample
from bt7 import session_table_crypto, FEE

TFS = ["1m", "3m", "5m", "15m", "30m", "1H", "4H"]
MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240}


def prep(m1):
    """ТФ з хвилинок (однакова історія для всіх ТФ). Повертає {tf: dict масивів}."""
    out = {}
    for tf in TFS:
        d = m1 if tf == "1m" else resample(m1, MIN[tf])
        w1, _ = bot.wavetrend(d)
        out[tf] = {"t": ((d.index + pd.Timedelta(minutes=MIN[tf])).values.astype("datetime64[m]").astype(np.int64)),
                   "o": d["open"].to_numpy(), "h": d["high"].to_numpy(), "l": d["low"].to_numpy(),
                   "c": d["close"].to_numpy(), "w": w1.to_numpy()}
    return out


class Side:
    """Сетап в один бік. Для шорта ціни дзеркалимо (×−1), тож логіка одна."""

    def __init__(self, sg):
        self.sg = sg
        self.reset()

    def reset(self):
        self.active = False
        self.dno = None
        self.base = None
        self.t0 = None
        self.zone_tfs = set()
        self.trig = None


class Engine:
    """Мультитаймфреймовий детектор сигналів для одного активу. Отримує закриті свічки по черзі (step).
    Сигнал: (sg, base, ftf, ціна, дно, [ТФ із дном], (низ FVG, верх FVG))."""

    def __init__(self, NEAR=1.5, ENTRY_TF="fvg", FROM="trig", MIN_BASE="1m", FRESH=False, RELAX_TF=None):
        """RELAX_TF: дно/вершина нижче MIN_BASE теж годиться, якщо закріп — за FVG цього ТФ або старшого."""
        self.NEAR, self.ENTRY_TF, self.FROM, self.MIN_BASE, self.FRESH = NEAR, ENTRY_TF, FROM, MIN_BASE, FRESH
        self.RELAX_TF = RELAX_TF
        self.S = {1: Side(1), -1: Side(-1)}
        # FVG проти: [низ, верх, межа закріпу, t утворення] у «лонгових» (для шорта — дзеркальних) координатах
        self.fv = {sg: {tf: [] for tf in TFS} for sg in (1, -1)}
        self.inzone = {sg: {tf: False for tf in TFS} for sg in (1, -1)}
        self.hist = {tf: {"h": [], "l": [], "w": []} for tf in TFS}   # останні свічки для «хвилі» дна
        self.last = {tf: None for tf in TFS}                           # час закриття останньої обробленої свічки
        self.dbg = None                                                # список для діагностики (mtfdebug)
        self.pf = {tf: None for tf in TFS}                             # свічка, що формується: [high, low] з хвилинок
        self.STRONG = "15m"                                            # «старший» FVG — від цього ТФ

    def step(self, t, closed, m1, atr):
        """t — час (хвилини від epoch) закриття; closed — {tf: (h, l, c, w)} свічок, що закрились у t;
        m1 — (h, l, c) хвилинки, що закрилась у t (або None); atr — ATR(M15). Повертає список сигналів."""
        for tf, (h, l, c, w) in closed.items():
            H = self.hist[tf]
            H["h"].append(h); H["l"].append(l); H["w"].append(w)
            if len(H["h"]) > 40:
                for key in H:
                    del H[key][0]
            self.last[tf] = t
        # 1) WT-зони (дно / базовий ТФ / ТФ тригера)
        for tf, (h, l, c, w) in closed.items():
            H = self.hist[tf]
            n = len(H["w"])
            for sg in (1, -1):
                st = self.S[sg]
                now_in = w * sg <= bot.WT_OS
                was = self.inzone[sg][tf]
                self.inzone[sg][tf] = now_in
                if now_in and not was:
                    ww = [x * sg for x in H["w"]]
                    j = n - 1
                    kk = j - 1
                    while kk > 0 and kk > j - 21 and ww[kk] < ww[kk - 1]:
                        kk -= 1
                    kk = max(kk, 0)
                    lows = H["l"][kk:j + 1] if sg == 1 else [-x for x in H["h"][kk:j + 1]]
                    d0 = float(min(lows))
                    others = any(self.inzone[sg][x] for x in TFS if x != tf)
                    # новий сетап: нічого не відкрито, або інші ТФ уже вийшли із зони, або нове дно вище за старе
                    # (нова вершина нижча). Інакше — те саме дно росте на старших ТФ (або оновлюється нижче)
                    if not st.active or (self.FRESH and (not others or d0 > st.dno)):
                        st.active, st.dno, st.base, st.t0, st.zone_tfs, st.trig = True, d0, tf, t, {tf}, tf
                    else:
                        st.dno = min(st.dno, d0)
                        st.zone_tfs.add(tf)
                        if TFS.index(tf) > TFS.index(st.base):
                            st.base = tf
                        if TFS.index(tf) < TFS.index(st.trig):
                            st.trig = tf
        # свічки, що формуються (з хвилинок): для «потенційних» FVG
        for tf in TFS[1:]:
            if tf in closed:
                self.pf[tf] = None
            elif m1 is not None and self.last[tf] is not None:
                p = self.pf[tf]
                self.pf[tf] = [m1[0], m1[1]] if p is None else [max(p[0], m1[0]), min(p[1], m1[1])]
        # 2) злам/зсув дна і закріп (FVG — у стані ДО цієї свічки)
        sigs = []
        for sg in (1, -1):
            st = self.S[sg]
            if not st.active:
                continue
            if m1 is not None:
                l1 = m1[1] if sg == 1 else -m1[0]
                c1 = m1[2] * sg
                if t - st.t0 > 3 * 24 * 60:
                    st.reset()
                    continue
                if c1 < st.dno:
                    if self.FRESH and any(self.inzone[sg][x] for x in TFS):
                        st.dno = l1             # WT ще в зоні — тіло нижче дна лише переносить дно (вершину)
                    else:
                        st.reset()
                        continue
                if l1 < st.dno:
                    st.dno = l1
            else:
                c1 = None
            low_base = TFS.index(st.base) < TFS.index(self.MIN_BASE)
            if low_base and not self.RELAX_TF:
                if self.dbg is not None:
                    self.dbg.append((t, sg, "база нижче мінімуму", dict(base=st.base, zones=sorted(st.zone_tfs, key=TFS.index),
                                                                       dno=st.dno * sg)))
                continue                                    # дно ще не підтвердилось на потрібному ТФ
            bi = TFS.index(st.base) if self.FROM == "base" else TFS.index(st.trig)
            if not any(tf in closed for tf in TFS[bi:]) or c1 is None or not atr or atr != atr:
                continue
            cands = []
            for tf in TFS[bi:]:
                for f in self.fv[sg][tf]:
                    if f[0] > st.dno and f[0] - c1 <= self.NEAR * atr:
                        cands.append((TFS.index(tf), tf, f))
            si = max(TFS.index(self.STRONG), TFS.index(st.base))
            if cands and self.FRESH:
                # старший FVG перебиває молодший там, де він перекриває найближчий FVG (а не далі по ходу)
                near = min(cands, key=lambda x: x[2][0])
                group = [x for x in cands if x[2][0] <= near[2][1] + 0.1 * atr]
                top = max(x[0] for x in group)
                need = min((x for x in group if x[0] == top), key=lambda x: x[2][0])
                if top < TFS.index(self.STRONG):
                    # найближчий FVG дрібний (M1–M5): якщо поруч є FVG M15+ (не молодший бази) — потрібен він
                    strong = [x for x in cands if x[0] >= si]
                    if strong:
                        top2 = max(x[0] for x in strong)
                        need = min((x for x in strong if x[0] == top2), key=lambda x: x[2][0])
            elif cands:
                top = max(x[0] for x in cands)
                need = min((x for x in cands if x[0] == top), key=lambda x: x[2][0])
            else:
                own = [f for f in self.fv[sg][TFS[bi]] if f[3] >= st.t0 and f[0] > st.dno]
                if not own:
                    continue
                need = (bi, TFS[bi], min(own, key=lambda f: f[0]))
            if self.FRESH and need[0] < TFS.index(self.STRONG):
                # потенційний FVG M15+: свічка ще формується, а розрив над ціною вже є — чекаємо її закриття
                pot = None
                for tf in TFS[si:]:
                    p = self.pf[tf]
                    H = self.hist[tf]
                    if p is None or len(H["h"]) < 2:
                        continue
                    ph = p[0] if sg == 1 else -p[1]
                    l2 = H["l"][-2] if sg == 1 else -H["h"][-2]
                    if ph < l2 and ph > st.dno and ph - c1 <= self.NEAR * atr:
                        pot = (tf, ph * sg, l2 * sg)
                if pot:
                    if self.dbg is not None:
                        self.dbg.append((t, sg, f"чекаю закриття {pot[0]} — потенційний FVG",
                                         dict(base=st.base, zones=sorted(st.zone_tfs, key=TFS.index), dno=st.dno * sg,
                                              pot=sorted(pot[1:]))))
                    continue
            if low_base and TFS.index(need[1]) < TFS.index(self.RELAX_TF):
                if self.dbg is not None:
                    self.dbg.append((t, sg, f"база {st.base} нижче {self.MIN_BASE}, а FVG лише {need[1]}",
                                     dict(base=st.base, zones=sorted(st.zone_tfs, key=TFS.index), dno=st.dno * sg,
                                          need_tf=need[1])))
                continue
            etf = need[1] if self.ENTRY_TF == "fvg" else st.base
            if self.dbg is not None:
                self.dbg.append((t, sg, "чекаю закріп" if etf in closed else "ТФ закріпу ще не закрився",
                                 dict(base=st.base, trig=st.trig, zones=sorted(st.zone_tfs, key=TFS.index),
                                      dno=st.dno * sg, need_tf=need[1],
                                      fvg=sorted((need[2][0] * sg, need[2][1] * sg)), edge=need[2][2] * sg,
                                      close=(closed[etf][2] if etf in closed else None),
                                      cands=[(x[1], round(x[2][0] * sg, 4), round(x[2][2] * sg, 4)) for x in cands][:8])))
            if etf not in closed:
                continue
            ce = closed[etf][2] * sg
            if ce > need[2][2] and self.FRESH:
                # закріп не має закриватися всередині FVG проти нас старшого ТФ (або ближче 0.25 ATR до нього):
                # тоді чекаємо закріпу і за ним
                blk = None
                for tf in TFS[need[0] + 1:]:
                    for f in self.fv[sg][tf]:
                        if f[0] - 0.25 * atr <= ce <= f[2] and f[0] > st.dno:
                            blk = (tf, f)
                if blk:
                    if self.dbg is not None:
                        self.dbg.append((t, sg, f"закріп усередині FVG {blk[0]} — чекаю закріп за ним",
                                         dict(base=st.base, dno=st.dno * sg, need_tf=need[1], close=ce * sg,
                                              blk=sorted((blk[1][0] * sg, blk[1][1] * sg)))))
                    continue
            if ce > need[2][2]:
                f = need[2]
                lo, hi = sorted((f[0] * sg, f[1] * sg))
                sigs.append((sg, st.base, need[1], float(ce * sg), float(st.dno * sg),
                             sorted(st.zone_tfs, key=TFS.index), (lo, hi)))
                st.reset()
        # 3) оновити FVG проти
        for tf, (h0, l0_, c0, w0) in closed.items():
            H = self.hist[tf]
            for sg in (1, -1):
                h = h0 if sg == 1 else -l0_
                c = c0 * sg
                keep = []
                for f in self.fv[sg][tf]:
                    if c > f[2]:
                        continue
                    if h >= f[0]:
                        f[2] = max(f[2], h)
                    keep.append(f)
                if len(H["h"]) >= 3:
                    l2 = H["l"][-3] if sg == 1 else -H["h"][-3]
                    if h < l2:
                        keep.append([h, l2, l2, t])
                self.fv[sg][tf] = keep[-60:]
        return sigs


def run_variant(D, atr_at, bias_at, NEAR, ENTRY_TF, EXIT, CTX, FROM="base", EXIT_REF="base", EXIT_OFF=0,
                MIN_BASE="1m", FRESH=False, RELAX_TF=None, XENG=False):
    """Бектест однієї конфігурації на рушії Engine. Одна позиція на актив.
    CTX: 1 — лише в бік контексту; 0 — без контексту; 2 — проти контексту лише з базою ≥ M15.
    XENG: реверсні виходи шукає окремий рушій за старими правилами (як у живому боті)."""
    eng = Engine(NEAR, ENTRY_TF, FROM, MIN_BASE, FRESH, RELAX_TF)
    xeng = Engine(NEAR, ENTRY_TF, FROM, MIN_BASE, False, None) if XENG else None
    ti = {tf: 0 for tf in TFS}
    tm = D["1m"]["t"]
    a1 = D["1m"]
    trades, pos = [], None
    for k in range(len(tm)):
        t = tm[k]
        closed = {}
        for tf in TFS:
            arr = D[tf]
            j = ti[tf]
            if j < len(arr["t"]) and arr["t"][j] <= t:
                ti[tf] = j + 1
                closed[tf] = (arr["h"][j], arr["l"][j], arr["c"][j], arr["w"][j])
        m1b = (a1["h"][k], a1["l"][k], a1["c"][k])
        at = atr_at(t)
        sig_now = eng.step(t, closed, m1b, at)
        x_now = xeng.step(t, closed, m1b, at) if xeng is not None else []
        if pos is not None:
            lo = a1["l"][k] if pos["sg"] == 1 else a1["h"][k]
            if (pos["sg"] == 1 and lo <= pos["stop"]) or (pos["sg"] == -1 and lo >= pos["stop"]):
                pos.update(exit=pos["stop"], r=-1.0, how="стоп", t1=int(t))
                trades.append(pos)
                pos = None
        if pos is not None:
            for sg, base, ftf, price, dno, _, _ in x_now + sig_now:
                ref = TFS.index(pos["base"] if EXIT_REF == "base" else pos["ftf"]) - EXIT_OFF
                if sg == -pos["sg"] and (EXIT == "any" or TFS.index(base) >= ref):
                    pos.update(exit=price, r=pos["sg"] * (price - pos["entry"]) / pos["R"], how="реверс", t1=int(t))
                    trades.append(pos)
                    pos = None
                    break
        if pos is not None and t - pos["t0"] > 10 * 24 * 60:
            price = a1["c"][k]
            pos.update(exit=price, r=pos["sg"] * (price - pos["entry"]) / pos["R"], how="час", t1=int(t))
            trades.append(pos)
            pos = None
        for sg, base, ftf, price, dno, _, _ in sig_now:
            if pos is not None:
                break
            b = bias_at(t)
            with_ctx = b == ("BULL" if sg == 1 else "BEAR")
            if CTX == 1 and not with_ctx:
                continue
            if CTX == 2 and not with_ctx and TFS.index(base) < TFS.index("15m"):
                continue
            stop = dno - sg * price * 0.0001
            R = sg * (price - stop)
            if R <= price * 0.0003:
                continue
            pos = {"sg": sg, "entry": price, "stop": stop, "R": R, "base": base, "ftf": ftf, "t0": int(t),
                   "ctx": with_ctx}
    return trades


def stats(rr):
    a = np.asarray(rr, float)
    if not len(a):
        return None
    w, ls = a[a > 0], a[a <= 0]
    eq = np.r_[0, np.cumsum(a)]
    return {"n": len(a), "win%": round(float(len(w) / len(a) * 100)), "sumR": round(float(a.sum()), 1),
            "PF": round(float(w.sum() / -ls.sum()), 2) if ls.sum() < 0 else None,
            "avgWin": round(float(w.mean()), 2) if len(w) else 0, "DD": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}


def run(src):
    now = pd.Timestamp.now(tz="UTC")
    okx = src if type(src).__name__ == "OkxSource" else getattr(src, "okx", None)
    # (назва, CTX, нова версія?)
    variants = [
        ("СТАРА J3 (23.11), лише в бік контексту", 1, False),
        ("ЖИВА (25.11), лише в бік контексту", 1, True),
        ("ЖИВА, без контексту", 0, True),
        ("ЖИВА, проти контексту лише з базою ≥ M15", 2, True),
    ]
    allres = {v[0]: [] for v in variants}
    for sym in bot.SYMBOLS:
        t0 = time.time()
        data, sname = load(sym, okx, now)
        m1 = data["1m"]
        D = prep(m1)
        h1 = resample(m1, 60)
        sess = session_table_crypto(h1)
        s_start = (sess["start"].values.astype("datetime64[m]").astype(np.int64))
        bias = sess["bias"].to_numpy()
        d15 = resample(m1, 15)
        atr15 = (d15["high"] - d15["low"]).rolling(14).mean().to_numpy()
        t15 = D["15m"]["t"]

        def atr_at(t):
            k = int(np.searchsorted(t15, t, side="right")) - 1
            return atr15[k] if k >= 14 else None

        def bias_at(t):
            k = int(np.searchsorted(s_start, t, side="right")) - 1
            return bias[k] if 0 <= k < len(bias) else None

        print(f"{sym}: хвилинок {len(m1)} з {m1.index[0]} по {m1.index[-1]}")
        for name, ctx, new in variants:
            if new:
                tr = run_variant(D, atr_at, bias_at, 1.5, "fvg", "base", ctx, "trig", "base", 0, "3m",
                                 True, "15m", True)
            else:
                tr = run_variant(D, atr_at, bias_at, 1.5, "fvg", "base", ctx, "trig", "base", 0, "3m")
            for x in tr:
                x["sym"] = sym
                x["rf"] = x["r"] - FEE * x["entry"] / x["R"]
            allres[name] += tr
        print(f"{sym}: {time.time() - t0:.0f} c")
    for name, tr in allres.items():
        tr.sort(key=lambda x: x["t0"])
        out = {"v": name, "noFee": stats([x["r"] for x in tr]), "fee": stats([x["rf"] for x in tr])}
        out["byBase"] = {tf: stats([x["rf"] for x in tr if x["base"] == tf]) for tf in TFS}
        out["byFvgTf"] = {tf: stats([x["rf"] for x in tr if x["ftf"] == tf]) for tf in TFS}
        out["bySym"] = {s: stats([x["rf"] for x in tr if x["sym"] == s]) for s in bot.SYMBOLS}
        out["byCtx"] = {"з контекстом": stats([x["rf"] for x in tr if x.get("ctx")]),
                        "проти": stats([x["rf"] for x in tr if x.get("ctx") is False])}
        out["how"] = {h: sum(1 for x in tr if x["how"] == h) for h in ("стоп", "реверс", "час")}
        out["feeR_avg"] = round(float(np.mean([x["r"] - x["rf"] for x in tr])), 3) if tr else None
        half = len(tr) // 2
        out["halves"] = [round(sum(x["rf"] for x in tr[:half]), 1), round(sum(x["rf"] for x in tr[half:]), 1)]
        print("MTF", json.dumps(out, ensure_ascii=False))
