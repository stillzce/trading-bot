"""Великий бектест (bt4): максимум історії й ТФ.
Входи: WT (вихід із зони, як у боті) і FVG (закріп за FVG після дна/вершини WT, за спекою).
Стоп: за дном/вершиною, з мінімумом k×ATR(14) M15, k ∈ {0, 0.5, 1, 1.5, 2}.
FVG-стіна: відстань до найближчого незаповненого FVG проти угоди (набір A = M15+H1, набір B = ТФ сигналу і всі старші).
Супровід: E1 — як у боті (½ на протилежному WT свого ТФ → БУ, решта на протилежному WT наступного ТФ);
          E4 — ½ фіксується на найближчому FVG проти (якщо ціна дійде туди раніше), далі як E1.
python bot.py bt4"""
import heapq
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import bot
from backtest import session_table
from bt3 import fvg_entries

TFS = ["1m", "2m", "3m", "5m", "15m", "30m", "1H", "2H", "4H"]
TFM = {"1m": 1, "2m": 2, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "2H": 120, "4H": 240, "1D": 1440}
EXIT = {"1m": "5m", "2m": "5m", "3m": "15m", "5m": "15m", "15m": "1H", "30m": "4H", "1H": "4H", "2H": "4H", "4H": "1D"}
HORIZON_D = {"1m": 2, "2m": 2, "3m": 2, "5m": 3, "15m": 5, "30m": 7, "1H": 10, "2H": 15, "4H": 25}
KS = [0.0, 0.5, 1.0, 1.5, 2.0]
COSTF = {"EURUSD": 0.00012, "GBPUSD": 0.00016, "AUDUSD": 0.00012, "NZDUSD": 0.00016}
OKX_DAYS = {"1m": 60, "5m": 180, "1H": 365}
YH_DAYS = {"1m": 28, "5m": 58, "1H": 365}


def cost_of(sym, price):
    return COSTF.get(sym, price * 0.0001)


# ---------------------------------------------------------------- дані
def yahoo(sym, interval, days):
    import yfinance as yf
    tk = bot.YahooSource.TICKERS.get(sym, sym)
    now = pd.Timestamp.now(tz="UTC")
    parts = []
    if interval == "1m":
        end = now
        start_lim = now - pd.Timedelta(days=days)
        while end > start_lim:
            start = max(start_lim, end - pd.Timedelta(days=7))
            df = yf.Ticker(tk).history(start=start.to_pydatetime(), end=end.to_pydatetime(), interval="1m",
                                       prepost=True, auto_adjust=False)
            parts.append(df)
            end = start
            time.sleep(0.5)
    else:
        df = yf.Ticker(tk).history(period=f"{days}d", interval=interval, prepost=True, auto_adjust=False)
        parts.append(df)
    df = pd.concat(parts)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df = df[~df.index.duplicated()].sort_index()
    return df


def resample(df, minutes):
    r = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return r


def load(sym, okx, now):
    """Повертає {tf: df} з максимально доступною історією."""
    out = {}
    if okx is not None and okx.has(sym):
        m1 = okx.bars(sym, "1m", now - pd.Timedelta(days=OKX_DAYS["1m"]))
        m5 = okx.bars(sym, "5m", now - pd.Timedelta(days=OKX_DAYS["5m"]))
        h1 = okx.bars(sym, "1H", now - pd.Timedelta(days=OKX_DAYS["1H"]))
        src = "OKX"
    else:
        m1 = yahoo(sym, "1m", YH_DAYS["1m"])
        m5 = yahoo(sym, "5m", YH_DAYS["5m"])
        h1 = yahoo(sym, "60m", YH_DAYS["1H"])
        src = "Yahoo"
    m1, m5, h1 = [d[["open", "high", "low", "close"]].astype(float) for d in (m1, m5, h1)]
    out["1m"] = m1
    for k in (2, 3):
        out[f"{k}m"] = resample(m1, k)
    out["5m"] = m5
    out["15m"] = resample(m5, 15)
    out["30m"] = resample(m5, 30)
    out["1H"] = h1
    out["2H"] = resample(h1, 120)
    out["4H"] = resample(h1, 240)
    out["1D"] = h1.resample("1D", offset="22h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    cut = now.floor("min")
    for tf, d in out.items():  # лише закриті свічки
        out[tf] = d[d.index + pd.Timedelta(minutes=TFM[tf]) <= cut]
    return out, src


# ---------------------------------------------------------------- FVG із часом заповнення (O(n log n))
def fvg_table(df, tf):
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    cl = (df.index + pd.Timedelta(minutes=TFM[tf])).values.astype("datetime64[ns]").astype(np.int64)
    rows = []  # dir(+1 bull/-1 bear), lo, hi, t_created, t_filled
    bear_heap, bull_heap = [], []  # bear: заповнюється close > hi (min-heap по hi); bull: close < lo (max-heap по lo)
    BIG = np.iinfo(np.int64).max
    for i in range(len(c)):
        while bear_heap and c[i] > bear_heap[0][0]:
            _, k = heapq.heappop(bear_heap)
            rows[k][4] = cl[i]
        while bull_heap and c[i] < -bull_heap[0][0]:
            _, k = heapq.heappop(bull_heap)
            rows[k][4] = cl[i]
        if i >= 2:
            if l[i] > h[i - 2]:
                rows.append([1, h[i - 2], l[i], cl[i], BIG])
                heapq.heappush(bull_heap, (-h[i - 2], len(rows) - 1))
            elif h[i] < l[i - 2]:
                rows.append([-1, h[i], l[i - 2], cl[i], BIG])
                heapq.heappush(bear_heap, (l[i - 2], len(rows) - 1))
    a = np.array(rows, dtype=object) if rows else np.zeros((0, 5), dtype=object)
    return {"dir": a[:, 0].astype(int) if len(a) else np.zeros(0, int),
            "lo": a[:, 1].astype(float) if len(a) else np.zeros(0),
            "hi": a[:, 2].astype(float) if len(a) else np.zeros(0),
            "tc": a[:, 3].astype(np.int64) if len(a) else np.zeros(0, np.int64),
            "tf": a[:, 4].astype(np.int64) if len(a) else np.zeros(0, np.int64)}


def nearest_against(tabs, t, side, entry):
    """Відстань у ціні до ближнього краю найближчого незаповненого FVG проти угоди (0 — ціна всередині)."""
    best = np.inf
    for f in tabs:
        if not len(f["dir"]):
            continue
        alive = (f["tc"] <= t) & (f["tf"] > t)
        if side == "LONG":
            m = alive & (f["dir"] == -1) & (f["hi"] > entry)
            if m.any():
                best = min(best, float(np.maximum(0.0, f["lo"][m] - entry).min()))
        else:
            m = alive & (f["dir"] == 1) & (f["lo"] < entry)
            if m.any():
                best = min(best, float(np.maximum(0.0, entry - f["hi"][m]).min()))
    return best


# ---------------------------------------------------------------- головний прогін
def run(src):
    now = pd.Timestamp.now(tz="UTC")
    okx = getattr(src, "okx", None)
    rows = []
    meta = {}
    for sym in bot.SYMBOLS:
        t0 = time.time()
        try:
            data, sname = load(sym, okx, now)
        except Exception as e:
            print(f"{sym}: дані: {e}")
            continue
        meta[sym] = {"src": sname, **{tf: [str(d.index[0])[:10], str(d.index[-1])[:10], len(d)] for tf, d in data.items() if len(d)}}
        sess = session_table(data["1H"])
        s_start = sess["start"].values.astype("datetime64[ns]").astype(np.int64)
        bias = sess["bias"].to_numpy()
        W, OPP, FV, CL = {}, {}, {}, {}
        for tf, d in data.items():
            if len(d) < 100:
                continue
            w1, w2 = bot.wavetrend(d)
            W[tf] = (w1, w2)
            CL[tf] = (d.index + pd.Timedelta(minutes=TFM[tf])).values.astype("datetime64[ns]").astype(np.int64)
            ev = {"LONG": [], "SHORT": []}
            for i in range(1, len(d)):
                s = bot.wt_extreme_cross(w1, w2, i)
                if s:
                    ev[s].append(i)
            OPP[tf] = {s: np.array(v, int) for s, v in ev.items()}
            if tf != "1D":
                FV[tf] = fvg_table(d, tf)
        # ATR(14) M15 з фолбеком H1/2
        d15, dh = data["15m"], data["1H"]
        atr15 = (d15["high"] - d15["low"]).rolling(14).mean().to_numpy()
        atrh = ((dh["high"] - dh["low"]).rolling(14).mean() / 2).to_numpy()

        def atr_at(t):
            k = np.searchsorted(CL["15m"], t, side="right") - 1
            if k >= 14 and not np.isnan(atr15[k]):
                return atr15[k]
            k = np.searchsorted(CL["1H"], t, side="right") - 1
            return atrh[k] if k >= 14 else np.nan

        for tf in TFS:
            if tf not in W or EXIT[tf] not in OPP:
                continue
            d = data[tf]
            h, l, c = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
            cl = CL[tf]
            w1, w2 = W[tf]
            xtf = EXIT[tf]
            xcl, xc = CL[xtf], data[xtf]["close"].to_numpy()
            setA = [FV[t] for t in ("15m", "1H") if t in FV]
            setB = [FV[t] for t in TFS[TFS.index(tf):] if t in FV]
            hor = np.int64(HORIZON_D[tf] * 86400 * 10 ** 9)
            # входи
            entries = []
            for i in range(60, len(d)):
                s = bot.wt_extreme_cross(w1, w2, i)
                if s:
                    entries.append(("WT", i, s, bot.zone_extreme(d, w1, i, s)))
            wv = w1.to_numpy()
            for i, dno, _ in fvg_entries(h, l, c, wv, bot.WT_OS):
                if i >= 60:
                    entries.append(("FVG", i, "LONG", dno))
            for i, dno, _ in fvg_entries(-l, -h, -c, -wv, -bot.WT_OB):
                if i >= 60:
                    entries.append(("FVG", i, "SHORT", -dno))
            for etype, i, side, ext in entries:
                t = cl[i]
                loc = pd.Timestamp(t, tz="UTC").tz_convert(bot.LOCAL_TZ)
                if loc.weekday() >= 5 or not (8 <= loc.hour < 23):
                    continue
                k = np.searchsorted(s_start, t, side="right") - 1
                if k < 0 or k >= len(bias):
                    continue
                b = bias[k]
                ctx = "with" if b == ("BULL" if side == "LONG" else "BEAR") else ("neutral" if b == "NEUTRAL" else "against")
                long_ = side == "LONG"
                sg = 1 if long_ else -1
                entry = c[i]
                cst = cost_of(sym, entry)
                atr = atr_at(t)
                if np.isnan(atr) or atr <= 0:
                    continue
                dA = nearest_against(setA, t, side, entry)
                dB = nearest_against(setB, t, side, entry)
                # події
                opp = "SHORT" if long_ else "LONG"
                oi = OPP[tf][opp]
                oi = oi[oi > i]
                jh = int(oi[0]) if len(oi) else None
                xo = OPP[xtf][opp]
                xt = xcl[xo]
                kx = np.searchsorted(xt, t, side="right")
                tx, px = (xt[kx], xc[xo[kx]]) if kx < len(xt) else (None, None)
                jx = int(np.searchsorted(cl, tx, side="left")) if tx is not None else None
                jend = int(np.searchsorted(cl, t + hor, side="right")) - 1
                jend = min(jend, len(c) - 1)
                if jend <= i:
                    continue
                hh, ll = h[i + 1:jend + 1], l[i + 1:jend + 1]
                fav = hh if long_ else ll  # для торкання цілі
                tp = entry + sg * dB if np.isfinite(dB) and dB > 0 else None
                res = {}
                for kk in KS:
                    stop = ext - sg * cst / 2
                    if kk > 0 and abs(entry - stop) < kk * atr:
                        stop = entry - sg * kk * atr
                    R = abs(entry - stop)
                    if R < 3 * cst or sg * (entry - stop) <= 0:
                        res[kk] = None
                        continue
                    hit = (ll <= stop) if long_ else (hh >= stop)
                    js = i + 1 + int(np.argmax(hit)) if hit.any() else None
                    out = {}
                    for ex in ("E1", "E4"):
                        jhh, phh = (jh, c[jh] if jh is not None else None)
                        if ex == "E4" and tp is not None:
                            touch = (fav >= tp) if long_ else (fav <= tp)
                            if touch.any():
                                jt = i + 1 + int(np.argmax(touch))
                                if jhh is None or jt < jhh:
                                    jhh, phh = jt, tp
                        big = 10 ** 12
                        a_s = js if js is not None and js <= jend else big
                        a_h = jhh if jhh is not None and jhh <= jend else big
                        a_x = jx if jx is not None and jx <= jend else big
                        if a_s <= min(a_h, a_x) and a_s < big:
                            r = sg * (stop - entry) / R
                        elif a_x < big and a_x <= a_h:
                            r = sg * (px - entry) / R
                        elif a_h < big:
                            r1 = sg * (phh - entry) / R
                            if r1 <= 0:
                                r = r1
                            else:
                                seg = l[a_h + 1:jend + 1] if long_ else h[a_h + 1:jend + 1]
                                be = (seg <= entry) if long_ else (seg >= entry)
                                jb = a_h + 1 + int(np.argmax(be)) if be.any() else big
                                if jb < a_x:
                                    r = r1 / 2
                                elif a_x < big:
                                    r = (r1 + sg * (px - entry) / R) / 2
                                else:
                                    r = (r1 + sg * (c[jend] - entry) / R) / 2
                        else:
                            r = sg * (c[jend] - entry) / R
                        out[ex] = round(r - cst / R, 3)
                    res[kk] = (round(R / atr, 3), out["E1"], out["E4"])
                rows.append((sym, tf, etype, side, ctx, loc.strftime("%Y-%m-%d %H:%M"),
                             round(dA / atr, 3) if np.isfinite(dA) else 99.0,
                             round(dB / atr, 3) if np.isfinite(dB) else 99.0, res))
        print(f"{sym} ({meta[sym]['src']}): записів {sum(1 for r in rows if r[0] == sym)} · {time.time() - t0:.0f} c")
    print("META", json.dumps(meta, ensure_ascii=False))
    report(rows)


def stats(a):
    a = np.asarray(a, float)
    if not len(a):
        return None
    pos, neg = a[a > 0].sum(), -a[a < 0].sum()
    eq = np.r_[0, np.cumsum(a)]
    return [len(a), round(float((a > 0).mean() * 100)), round(float(a.sum()), 1),
            round(float(pos / neg), 2) if neg else None, round(float((np.maximum.accumulate(eq) - eq).max()), 1),
            round(float(a.mean()), 3)]


WALLS = {"none": lambda r, Rk: True, "A0.5": lambda r, Rk: r[6] >= 0.5, "A1": lambda r, Rk: r[6] >= 1.0,
         "A1.5": lambda r, Rk: r[6] >= 1.5, "A2": lambda r, Rk: r[6] >= 2.0,
         "B0.5": lambda r, Rk: r[7] >= 0.5, "B1": lambda r, Rk: r[7] >= 1.0, "B1.5": lambda r, Rk: r[7] >= 1.5,
         "R1(B)": lambda r, Rk: r[7] >= Rk, "R2(B)": lambda r, Rk: r[7] >= 2 * Rk}
CTXS = {"all": lambda c: True, "with": lambda c: c == "with", "against": lambda c: c == "against",
        "neutral": lambda c: c == "neutral"}


def report(rows):
    rows.sort(key=lambda r: r[5])
    print("ROWS", len(rows))
    # 1) сітка
    for et in ("WT", "FVG"):
        for kk in KS:
            for wn, wf in WALLS.items():
                for cn, cf in CTXS.items():
                    for ei, ex in ((1, "E1"), (2, "E4")):
                        a = [r[8][kk][ei] for r in rows if r[2] == et and r[8].get(kk) and cf(r[4]) and wf(r, r[8][kk][0])]
                        st = stats(a)
                        if st:
                            print("GRID", json.dumps([et, kk, wn, cn, ex] + st, ensure_ascii=False))
    # 2) розбивки для ключових конфігурацій
    for et in ("WT", "FVG"):
        for kk in (0.0, 1.0, 1.5):
            for wn in ("none", "A1", "B1"):
                for cn in ("all", "with", "against"):
                    wf, cf = WALLS[wn], CTXS[cn]
                    sel = [r for r in rows if r[2] == et and r[8].get(kk) and cf(r[4]) and wf(r, r[8][kk][0])]
                    by = defaultdict(lambda: defaultdict(list))
                    mid = sel[len(sel) // 2][5] if sel else ""
                    for r in sel:
                        v = r[8][kk][1]
                        by["tf"][r[1]].append(v)
                        by["sym"][r[0]].append(v)
                        by["month"][r[5][:7]].append(v)
                        by["half"]["1" if r[5] < mid else "2"].append(v)
                        by["tf_month"][f"{r[1]}|{r[5][:7]}"].append(v)
                        by["sym_tf"][f"{r[0]}|{r[1]}"].append(v)
                        by["side"][r[3]].append(v)
                        by["Rsize"][str(min(4, int(r[8][kk][0] * 2) / 2))].append(v)
                    out = {g: {k: stats(v) for k, v in d.items()} for g, d in by.items()}
                    print("BY", json.dumps({"cfg": [et, kk, wn, cn, "E1"], **out}, ensure_ascii=False))
