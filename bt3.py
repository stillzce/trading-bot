"""Бектест входу за специфікацією WaveTrend + FVG: дно/вершина з WT (зі свіпами й інвалідацією),
вхід — тіло свічки закрилось за найближчим незаповненим FVG проти нас (з урахуванням тіней), на ТФ дна.
python bot.py bt3[:днів]"""
import json, os
import numpy as np, pandas as pd
import bot
from backtest import session_table

DAYS = int(os.environ.get("BT_DAYS", "45"))
TF_DAYS = {"1m": 7, "5m": DAYS, "15m": DAYS}
NEXT = {"1m": "5m", "5m": "15m", "15m": "1H", "1H": "4H"}
COSTF = {"EURUSD": 0.00012, "GBPUSD": 0.00016, "AUDUSD": 0.00012, "NZDUSD": 0.00016}


def cost_of(sym, price):
    return COSTF.get(sym, price * 0.0001)


def fvg_entries(h, l, c, w1, os_):
    """Лонги на рядах (для шорта — дзеркальні ряди). [(i, дно, бар початку зони)]"""
    out = []
    active, dno, zs, used = False, np.nan, 0, False
    fvgs = []  # [низ, верх, межа, бар]
    for i in range(2, len(c)):
        if w1[i] <= os_:
            if not active or w1[i - 1] > os_:
                active, dno, zs, used = True, l[i], i, False
            else:
                dno = min(dno, l[i])
        elif active:
            if c[i] < dno:
                active = False
            elif l[i] < dno:
                dno = l[i]
        if active and not used:
            cands = [f for f in fvgs if f[3] >= zs - 3 and f[0] > dno]
            if cands:
                near = min(cands, key=lambda f: f[0])
                if c[i] > near[2]:
                    out.append((i, dno, zs))
                    used = True
        fvgs = [f for f in fvgs if c[i] <= f[2]]
        for f in fvgs:
            if h[i] >= f[0]:
                f[2] = max(f[2], h[i])
        if h[i] < l[i - 2]:
            fvgs.append([h[i], l[i - 2], l[i - 2], i])
    return out


def signals(df, w1):
    h, l, c, w = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), w1.to_numpy()
    longs = fvg_entries(h, l, c, w, bot.WT_OS)
    shorts = [(i, -d, z) for i, d, z in fvg_entries(-l, -h, -c, -w, -bot.WT_OB)]
    return {"LONG": longs, "SHORT": shorts}


def fvg_list(df, tf):
    dur = pd.Timedelta(minutes=bot.TF_MIN[tf])
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    closes = (df.index + dur).values
    out = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            d, lo, hi = "bull", h[i - 2], l[i]
        elif h[i] < l[i - 2]:
            d, lo, hi = "bear", h[i], l[i - 2]
        else:
            continue
        tz = None
        for j in range(i + 1, len(df)):
            if (d == "bull" and c[j] < lo) or (d == "bear" and c[j] > hi):
                tz = closes[j]
                break
        out.append((d, lo, hi, closes[i], tz))
    return out


def room_R(fvgs_by_tf, t, side, entry, R):
    """Відстань (у R) до найближчого незаповненого FVG проти угоди на будь-якому з ТФ."""
    best = np.inf
    for fv in fvgs_by_tf:
        for d, lo, hi, tc, tz in fv:
            if tc > t or (tz is not None and tz <= t):
                continue
            if side == "LONG" and d == "bear" and lo > entry:
                best = min(best, (lo - entry) / R)
            if side == "SHORT" and d == "bull" and hi < entry:
                best = min(best, (entry - hi) / R)
    return best


def run(src):
    now = pd.Timestamp.now(tz="UTC")
    trades = []
    today = str(now.tz_convert(bot.LOCAL_TZ).date())
    for sym in bot.SYMBOLS:
        try:
            if hasattr(src, "min_tf") and src.min_tf(sym) > 1:
                continue
            h1 = bot.closed_only(src.bars(sym, "1H", now - pd.Timedelta(days=DAYS + 40)), "1H")
            sess = session_table(h1)
            data = {"1H": h1, "4H": bot.closed_only(src.bars(sym, "4H", now - pd.Timedelta(days=DAYS + 60)), "4H")}
            for tf in ("1m", "5m", "15m"):
                data[tf] = bot.closed_only(src.bars(sym, tf, now - pd.Timedelta(days=TF_DAYS[tf] + 3)), tf)
        except Exception as e:
            print(f"{sym}: {e}")
            continue
        W = {tf: bot.wavetrend(df) for tf, df in data.items()}
        SIG = {tf: signals(data[tf], W[tf][0]) for tf in data}
        FV = {tf: fvg_list(data[tf], tf) for tf in data}
        d15 = data["15m"]
        a15 = (d15["high"] - d15["low"]).rolling(14).mean().to_numpy()
        c15 = (d15.index + pd.Timedelta(minutes=15)).values
        CL = {tf: (data[tf].index + pd.Timedelta(minutes=bot.TF_MIN[tf])).values for tf in data}

        def first_exit(tf, side, t0):
            """перший дзеркальний сигнал (вхід у зворотний бік) на tf, чия зона почалась після t0"""
            opp = "SHORT" if side == "LONG" else "LONG"
            cl = CL[tf]
            idx = data[tf].index.values
            for i, _, zs in SIG[tf][opp]:
                if idx[zs] >= t0 and cl[i] > t0:
                    return cl[i], float(data[tf]["close"].iloc[i])
            return None

        def first_top(tf, side, t0):
            w = W[tf][0].to_numpy()
            cl = CL[tf]
            k = np.searchsorted(cl, t0, side="right")
            for j in range(k, len(w)):
                if (w[j] >= bot.WT_OB) if side == "LONG" else (w[j] <= bot.WT_OS):
                    return cl[j]
            return None

        def opp_wt(tf, side, t0):
            w1, w2 = W[tf]
            opp = "SHORT" if side == "LONG" else "LONG"
            cl = CL[tf]
            k = np.searchsorted(cl, t0, side="right")
            for j in range(max(k, 1), len(w1)):
                if bot.wt_extreme_cross(w1, w2, j) == opp:
                    return cl[j], float(data[tf]["close"].iloc[j])
            return None

        for tf in ("1m", "5m", "15m", "1H"):
            df = data[tf]
            h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
            cl = CL[tf]
            since = (now - pd.Timedelta(days=TF_DAYS.get(tf, DAYS))).to_datetime64()
            nxt = NEXT[tf]
            for side in ("LONG", "SHORT"):
                sg = 1 if side == "LONG" else -1
                for i, dno, zs in SIG[tf][side]:
                    t0 = cl[i]
                    if t0 < since:
                        continue
                    loc = pd.Timestamp(t0).tz_localize("UTC").tz_convert(bot.LOCAL_TZ)
                    if loc.weekday() >= 5 or not (8 <= loc.hour < 23):
                        continue
                    k = np.searchsorted(sess["start"].values, t0, side="right") - 1
                    if k < 0 or k >= len(sess):
                        continue
                    bias = sess.iloc[k]["bias"]
                    ctx = "with" if bias == ("BULL" if side == "LONG" else "BEAR") else ("neutral" if bias == "NEUTRAL" else "against")
                    entry = c[i]
                    cst = cost_of(sym, entry)
                    kq = np.searchsorted(c15, t0, side="right") - 1
                    atr = a15[kq] if kq >= 14 else np.nan
                    for sv in ("spec", "atr1.5"):
                        stop = dno - sg * cst / 2
                        if sv == "atr1.5" and not np.isnan(atr) and abs(entry - stop) < 1.5 * atr:
                            stop = entry - sg * 1.5 * atr
                        R = abs(entry - stop)
                        if R < 3 * cst or sg * (entry - stop) <= 0:
                            continue
                        room = room_R([FV[tf], FV["15m"], FV["1H"]], t0, side, entry, R)
                        wall = (not np.isnan(atr)) and room_R([FV["15m"], FV["1H"]], t0, side, entry, atr) < 1.0
                        horizon = t0 + np.timedelta64(3, "D")
                        res = {}
                        # E1: як у боті зараз (½ на протилежному WT свого ТФ, решта — наступного ТФ, БУ після ½)
                        # E2: за спекою — БУ на 1R, вихід на дзеркальному FVG-сигналі ТФ входу
                        # E3: за спекою з драбиною — якщо на наступному ТФ сформувалась вершина раніше, виходимо за ним
                        ex2 = first_exit(tf, side, t0)
                        topN = first_top(nxt, side, t0)
                        ex3 = ex2
                        if topN is not None and (ex2 is None or topN < ex2[0]):
                            ex3 = first_exit(nxt, side, t0) or ex2
                        for name in ("E1", "E2", "E3"):
                            cur, half, be_done, out = stop, None, False, None
                            if name == "E1":
                                hx, fx = opp_wt(tf, side, t0), opp_wt(nxt, side, t0)
                            for j in range(i + 1, len(c)):
                                bt = cl[j]
                                if bt > horizon:
                                    break
                                hit = (l[j] <= cur) if side == "LONG" else (h[j] >= cur)
                                if name == "E1":
                                    if fx and fx[0] <= bt and not hit:
                                        r = sg * (fx[1] - entry) / R
                                        out = (half + r) / 2 if half is not None else r
                                        break
                                    if hit:
                                        out = half / 2 if half is not None else sg * (cur - entry) / R
                                        break
                                    if half is None and hx and hx[0] <= bt:
                                        r = sg * (hx[1] - entry) / R
                                        if r <= 0:
                                            out = r
                                            break
                                        half, cur = r, entry
                                else:
                                    ex = ex2 if name == "E2" else ex3
                                    if ex and ex[0] <= bt and not hit:
                                        out = sg * (ex[1] - entry) / R
                                        break
                                    if hit:
                                        out = sg * (cur - entry) / R
                                        break
                                    if not be_done and sg * ((h[j] if side == "LONG" else l[j]) - entry) >= R:
                                        cur, be_done = entry, True
                            if out is None:
                                jj = min(j, len(c) - 1)
                                r = sg * (c[jj] - entry) / R
                                out = (half + r) / 2 if (name == "E1" and half is not None) else r
                            res[name] = round(out - cst / R, 3)
                        trades.append(dict(sym=sym, tf=tf, side=side, t=str(loc)[:16], sv=sv, ctx=ctx,
                                           room=round(float(room), 2) if np.isfinite(room) else 99, wall=bool(wall),
                                           entry=float(entry), stop=float(stop), **res))
        print(f"{sym}: угод {len([x for x in trades if x['sym'] == sym])}")

    def st(rs):
        a = np.array(rs)
        if not len(a):
            return "-"
        pos, neg = a[a > 0].sum(), -a[a < 0].sum()
        eq = np.r_[0, np.cumsum(a)]
        return (f"n{len(a)} w{(a > 0).mean() * 100:.0f}% {a.sum():+.1f}R PF{pos / neg if neg else 0:.2f} "
                f"DD{(np.maximum.accumulate(eq) - eq).max():.1f}")
    mid = str((now - pd.Timedelta(days=DAYS / 2)).tz_convert(bot.LOCAL_TZ))[:16]
    filt = {
        "контекст: у бік": lambda x: x["ctx"] == "with",
        "контекст: будь-який": lambda x: True,
        "контекст: проти": lambda x: x["ctx"] == "against",
        "у бік + фільтр 1R": lambda x: x["ctx"] == "with" and x["room"] >= 1,
        "будь-який + фільтр 1R": lambda x: x["room"] >= 1,
        "у бік + без стіни (1 ATR)": lambda x: x["ctx"] == "with" and not x["wall"],
        "будь-який + без стіни (1 ATR)": lambda x: not x["wall"],
        "проти + без стіни (1 ATR)": lambda x: x["ctx"] == "against" and not x["wall"],
    }
    for sv in ("spec", "atr1.5"):
        for ex in ("E1", "E3"):
            for fn, f in filt.items():
                tt = sorted([x for x in trades if x["sv"] == sv and f(x)], key=lambda x: x["t"])
                row = {"all": st([x[ex] for x in tt])}
                for tf in ("1m", "5m", "15m", "1H"):
                    row[tf] = st([x[ex] for x in tt if x["tf"] == tf])
                row["5-15 1st"] = st([x[ex] for x in tt if x["tf"] != "1m" and x["t"] < mid])
                row["5-15 2nd"] = st([x[ex] for x in tt if x["tf"] != "1m" and x["t"] >= mid])
                print("BT3", f"{sv}|{ex}|{fn}", json.dumps(row, ensure_ascii=False))
    for x in trades:
        if x["t"][:10] >= str((now - pd.Timedelta(days=1)).tz_convert(bot.LOCAL_TZ).date()) and x["sv"] == "spec":
            print("TODAY", json.dumps(x, ensure_ascii=False))
