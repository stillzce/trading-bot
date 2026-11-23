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
    """Сетап і позиція в один бік. Для шорта ціни дзеркалимо (×−1), тож логіка одна."""

    def __init__(self, sg):
        self.sg = sg
        self.reset()

    def reset(self):
        self.active = False
        self.dno = None
        self.base = None
        self.t0 = None
        self.zone_tfs = set()


def run_variant(D, atr_at, bias_at, NEAR, ENTRY_TF, EXIT, CTX):
    """Повертає список угод."""
    ti = {tf: 0 for tf in TFS}            # скільки свічок кожного ТФ уже закрито
    tm = D["1m"]["t"]
    trades = []
    pos = None                            # відкрита позиція
    S = {1: Side(1), -1: Side(-1)}
    # FVG проти: для лонга — ведмежі (h[i] < l[i-2]), для шорта — бичачі. [низ, верх, межа закріпу, t утворення]
    fv = {sg: {tf: [] for tf in TFS} for sg in (1, -1)}
    inzone = {sg: {tf: False for tf in TFS} for sg in (1, -1)}
    for k in range(len(tm)):
        t = tm[k]
        closed = []
        for tf in TFS:
            arr = D[tf]
            j = ti[tf]
            if j < len(arr["t"]) and arr["t"][j] <= t:
                ti[tf] = j + 1
                closed.append((tf, j))
        closed_set = {tf for tf, _ in closed}
        a1 = D["1m"]
        # 1) WT-зони на закритих свічках (дно / базовий ТФ)
        for tf, j in closed:
            a = D[tf]
            for sg in (1, -1):
                st = S[sg]
                w = a["w"][j] * sg
                was = inzone[sg][tf]
                now_in = w <= bot.WT_OS
                inzone[sg][tf] = now_in
                if now_in and not was:
                    ww = a["w"][:j + 1] * sg
                    kk = j - 1
                    while kk > 0 and kk > j - 21 and ww[kk] < ww[kk - 1]:
                        kk -= 1
                    lows = (a["l"][kk:j + 1] if sg == 1 else -a["h"][kk:j + 1])
                    d0 = float(lows.min())
                    if not st.active:
                        st.active, st.dno, st.base, st.t0, st.zone_tfs = True, d0, tf, t, {tf}
                    else:
                        st.dno = min(st.dno, d0)
                        st.zone_tfs.add(tf)
                        if TFS.index(tf) > TFS.index(st.base):
                            st.base = tf
        # 2) злам/зсув дна і перевірка закріпу (FVG — у стані ДО цієї свічки, як у боті)
        sig_now = []
        for sg in (1, -1):
            st = S[sg]
            if not st.active:
                continue
            l1 = a1["l"][k] if sg == 1 else -a1["h"][k]
            c1 = a1["c"][k] * sg
            if c1 < st.dno or t - st.t0 > 3 * 24 * 60:
                st.reset()
                continue
            if l1 < st.dno:
                st.dno = l1
            bi = TFS.index(st.base)
            if not any(tf in closed_set for tf in TFS[bi:]):
                continue
            atr = atr_at(t)
            if not atr or np.isnan(atr):
                continue
            cands = []
            for tf in TFS[bi:]:
                for f in fv[sg][tf]:
                    if f[0] > st.dno and f[0] - c1 <= NEAR * atr:
                        cands.append((TFS.index(tf), tf, f))
            if cands:
                top = max(x[0] for x in cands)
                need = min((x for x in cands if x[0] == top), key=lambda x: x[2][0])
            else:
                own = [f for f in fv[sg][st.base] if f[3] >= st.t0 and f[0] > st.dno]
                if not own:
                    continue
                need = (bi, st.base, min(own, key=lambda f: f[0]))
            etf = need[1] if ENTRY_TF == "fvg" else st.base
            if etf not in closed_set:
                continue
            ce = D[etf]["c"][ti[etf] - 1] * sg
            if ce > need[2][2]:
                sig_now.append((sg, st.base, need[1], float(ce * sg), float(st.dno * sg)))
                st.reset()
        # 3) оновити FVG проти на закритих свічках
        for tf, j in closed:
            a = D[tf]
            for sg in (1, -1):
                h = a["h"][j] if sg == 1 else -a["l"][j]
                c = a["c"][j] * sg
                keep = []
                for f in fv[sg][tf]:
                    if c > f[2]:
                        continue
                    if h >= f[0]:
                        f[2] = max(f[2], h)
                    keep.append(f)
                if j >= 2:
                    l0 = a["l"][j - 2] if sg == 1 else -a["h"][j - 2]
                    if h < l0:
                        keep.append([h, l0, l0, t])
                fv[sg][tf] = keep[-60:]
        # 4) позиція: спершу стоп (усередині хвилини), потім реверс на закритті, потім час
        if pos is not None:
            lo = a1["l"][k] if pos["sg"] == 1 else a1["h"][k]
            if (pos["sg"] == 1 and lo <= pos["stop"]) or (pos["sg"] == -1 and lo >= pos["stop"]):
                pos.update(exit=pos["stop"], r=-1.0, how="стоп", t1=int(t))
                trades.append(pos)
                pos = None
        if pos is not None:
            for sg, base, ftf, price, dno in sig_now:
                if sg == -pos["sg"] and (EXIT == "any" or TFS.index(base) >= TFS.index(pos["base"])):
                    pos.update(exit=price, r=pos["sg"] * (price - pos["entry"]) / pos["R"], how="реверс", t1=int(t))
                    trades.append(pos)
                    pos = None
                    break
        if pos is not None and t - pos["t0"] > 10 * 24 * 60:
            price = a1["c"][k]
            pos.update(exit=price, r=pos["sg"] * (price - pos["entry"]) / pos["R"], how="час", t1=int(t))
            trades.append(pos)
            pos = None
        # 5) вхід
        for sg, base, ftf, price, dno in sig_now:
            if pos is not None:
                break
            b = bias_at(t)
            if CTX and b != ("BULL" if sg == 1 else "BEAR"):
                continue
            stop = dno - sg * price * 0.0001
            R = sg * (price - stop)
            if R <= price * 0.0003:
                continue
            pos = {"sg": sg, "entry": price, "stop": stop, "R": R, "base": base, "ftf": ftf, "t0": int(t)}
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
    variants = [
        ("A: закріп на ТФ FVG · вихід будь-який реверс · поруч 1.5 ATR", 1.5, "fvg", "any", 1),
        ("B: закріп на базовому ТФ · будь-який реверс · 1.5 ATR", 1.5, "base", "any", 1),
        ("C: як A, вихід лише реверс ≥ базового ТФ", 1.5, "fvg", "base", 1),
        ("D: як A, поруч 3 ATR", 3.0, "fvg", "any", 1),
        ("E: як C, поруч 3 ATR", 3.0, "fvg", "base", 1),
        ("F: як C, без фільтра контексту", 1.5, "fvg", "base", 0),
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
        for name, near, etf, ex, ctx in variants:
            tr = run_variant(D, atr_at, bias_at, near, etf, ex, ctx)
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
        out["how"] = {h: sum(1 for x in tr if x["how"] == h) for h in ("стоп", "реверс", "час")}
        out["feeR_avg"] = round(float(np.mean([x["r"] - x["rf"] for x in tr])), 3) if tr else None
        half = len(tr) // 2
        out["halves"] = [round(sum(x["rf"] for x in tr[:half]), 1), round(sum(x["rf"] for x in tr[half:]), 1)]
        print("MTF", json.dumps(out, ensure_ascii=False))
