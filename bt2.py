"""Бектест v2: сигнали як у боті (вихід WT із зони у бік контексту, 08–23 Київ), супровід як у боті
(½ на протилежному сигналі свого ТФ → стоп у БУ, решта на протилежному сигналі наступного ТФ).
Порівнює фільтри за старшими ТФ (H1/M15/H4) і варіанти стопу. python bot.py bt2[:днів]"""
import json, os
import numpy as np, pandas as pd
import bot
from backtest import session_table

DAYS = int(os.environ.get("BT_DAYS", "45"))
TF_DAYS = {"1m": 7, "5m": DAYS, "15m": DAYS}
COSTF = {"EURUSD": 0.00012, "GBPUSD": 0.00016, "AUDUSD": 0.00012, "NZDUSD": 0.00016}
NEXT = {"1m": "5m", "5m": "15m", "15m": "1H"}


def cost_of(sym, price):
    return COSTF.get(sym, price * 0.0001)


def htf_at(df, w1, times):
    """для кожного часу: WT останньої закритої свічки старшого ТФ і зміна до попередньої"""
    closes = (df.index + pd.Timedelta(minutes=bot.TF_MIN[df.attrs["tf"]])).values
    k = np.searchsorted(closes, times, side="right") - 1
    v = np.where(k >= 1, w1[np.clip(k, 0, None)], np.nan)
    d = np.where(k >= 1, w1[np.clip(k, 0, None)] - w1[np.clip(k - 1, 0, None)], np.nan)
    return v, d


def fvg_list(df, tf):
    """FVG на ТФ: (напрям, низ, верх, час появи, час закріпу проти, час повного заповнення).
    Бичачий: low[i] > high[i-2]. Ведмежий: high[i] < low[i-2]. Закріп проти = тіло закрилось за дальнім краєм."""
    dur = pd.Timedelta(minutes=bot.TF_MIN[tf])
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    closes = (df.index + dur).values
    out = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            out.append(["bull", h[i - 2], l[i], closes[i], None, None, i])
        elif h[i] < l[i - 2]:
            out.append(["bear", h[i], l[i - 2], closes[i], None, None, i])
    for f in out:
        i0 = f[6]
        for j in range(i0 + 1, len(df)):
            if f[0] == "bull":
                if f[5] is None and l[j] <= f[1]:
                    f[5] = closes[j]
                if c[j] < f[1]:
                    f[4] = closes[j]
                    break
            else:
                if f[5] is None and h[j] >= f[2]:
                    f[5] = closes[j]
                if c[j] > f[2]:
                    f[4] = closes[j]
                    break
    return out


def fvg_flags(fvgs, t, side, entry, atr, look):
    """Для угоди side у час t:
    zakrip — за останні look ціна закріпилась за FVG ПРОТИ угоди (для шорта: тіло вище ведмежого FVG);
    wall   — поруч у бік угоди стоїть незаповнений FVG протилежного напрямку (для шорта: бичачий FVG під ціною
             в межах 1 ATR M15 або ціна всередині нього)."""
    t = np.datetime64(t)
    zak, wall = False, False
    last_t, last_d = None, None
    for d, lo, hi, tc, tz, tf_, _ in fvgs:
        if tc > t:
            continue
        if tz is not None and tz <= t and (last_t is None or tz > last_t):
            last_t, last_d = tz, d
        if side == "SHORT":
            if d == "bear" and tz is not None and t - look < tz <= t:
                zak = True
            if d == "bull" and (tz is None or tz > t) and (tf_ is None or tf_ > t) and lo < entry and entry - hi <= atr:
                wall = True
        else:
            if d == "bull" and tz is not None and t - look < tz <= t:
                zak = True
            if d == "bear" and (tz is None or tz > t) and (tf_ is None or tf_ > t) and hi > entry and lo - entry <= atr:
                wall = True
    # last: останній закріп за FVG на цьому ТФ був ПРОТИ угоди (для шорта — тіло вище ведмежого FVG)
    last = last_d == ("bear" if side == "SHORT" else "bull")
    return zak, wall, last


def opp_times(df, w1, w2, side):
    opp = "SHORT" if side == "LONG" else "LONG"
    dur = pd.Timedelta(minutes=bot.TF_MIN[df.attrs["tf"]])
    out = []
    for i in range(1, len(df)):
        if bot.wt_extreme_cross(w1, w2, i) == opp:
            out.append((df.index[i] + dur, float(df["close"].iloc[i])))
    return out


def run(src):
    now = pd.Timestamp.now(tz="UTC")
    trades = []
    for sym in bot.SYMBOLS:
        try:
            if hasattr(src, "min_tf") and src.min_tf(sym) > 1:
                continue
            h1 = src.bars(sym, "1H", now - pd.Timedelta(days=DAYS + 40))
            sess = session_table(h1)
            data = {}
            for tf in ("1m", "5m", "15m", "1H", "4H"):
                days = TF_DAYS.get(tf, DAYS + 30)
                df = h1 if tf == "1H" else bot.closed_only(src.bars(sym, tf, now - pd.Timedelta(days=days + 3)), tf)
                df = bot.closed_only(df, tf)
                df.attrs["tf"] = tf
                w1, w2 = bot.wavetrend(df)
                data[tf] = (df, w1, w2)
        except Exception as e:
            print(f"{sym}: {e}")
            continue
        fv = {t_: fvg_list(data[t_][0], t_) for t_ in ("15m", "1H")}
        d15 = data["15m"][0]
        a15all = (d15["high"] - d15["low"]).rolling(14).mean()
        c15 = (d15.index + pd.Timedelta(minutes=15)).values
        for tf in ("1m", "5m", "15m"):
            df, w1, w2 = data[tf]
            if len(df) < 200:
                continue
            dur = pd.Timedelta(minutes=bot.TF_MIN[tf])
            since = now - pd.Timedelta(days=TF_DAYS[tf])
            closes = df.index + dur
            h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
            sig = [(i, bot.wt_extreme_cross(w1, w2, i)) for i in range(60, len(df)) if closes[i] >= since]
            sig = [(i, s) for i, s in sig if s]
            if not sig:
                continue
            times = np.array([closes[i].to_datetime64() for i, _ in sig])
            feats = {t: htf_at(data[t][0], data[t][1].to_numpy(), times) for t in ("15m", "1H", "4H")}
            opp_cache = {}
            for n, (i, side) in enumerate(sig):
                t = closes[i]
                loc = t.tz_convert(bot.LOCAL_TZ)
                if loc.weekday() >= 5 or not (8 <= loc.hour < 23):
                    continue
                k = np.searchsorted(sess["start"].values, t.to_datetime64(), side="right") - 1
                if k < 0 or k >= len(sess):
                    continue
                s = sess.iloc[k]
                if s["bias"] != ("BULL" if side == "LONG" else "BEAR"):
                    continue
                long_ = side == "LONG"
                sg = 1 if long_ else -1
                ext = bot.zone_extreme(df, w1, i, side)
                entry = c[i]
                cst = cost_of(sym, entry)
                for stopv in ("zone", "atr1.5"):
                    if stopv == "zone":
                        stop = ext - sg * cst / 2
                    else:  # ширший: не ближче ніж 1.5×ATR(14) M15
                        d15 = data["15m"][0]
                        a15 = (d15["high"] - d15["low"]).rolling(14).mean()
                        kk = np.searchsorted((d15.index + pd.Timedelta(minutes=15)).values, t.to_datetime64(), side="right") - 1
                        atr = float(a15.iloc[kk]) if kk >= 14 else np.nan
                        stop = ext - sg * cst / 2
                        mult = float(stopv[3:])
                        if not np.isnan(atr) and abs(entry - stop) < mult * atr:
                            stop = entry - sg * mult * atr
                    R = abs(entry - stop)
                    if R < 3 * cst or sg * (entry - stop) <= 0:
                        continue
                    # події
                    key = (tf, side)
                    if key not in opp_cache:
                        opp_cache[key] = opp_times(df, w1, w2, side)
                    nk = (NEXT[tf], side)
                    if nk not in opp_cache:
                        d2 = data[NEXT[tf]]
                        opp_cache[nk] = opp_times(d2[0], d2[1], d2[2], side)
                    half_t = next((x for x in opp_cache[key] if x[0] > t), None)
                    exit_t = next((x for x in opp_cache[nk] if x[0] > t), None)
                    horizon = t + pd.Timedelta(days=2)
                    res, half, cur = None, None, stop
                    for j in range(i + 1, len(df)):
                        bt = closes[j]
                        if bt > horizon:
                            break
                        hit = (l[j] <= cur) if long_ else (h[j] >= cur)
                        if exit_t and exit_t[0] <= bt and (not hit or exit_t[0] < bt):
                            r = sg * (exit_t[1] - entry) / R
                            res = (half + r) / 2 if half is not None else r
                            break
                        if hit:
                            res = half / 2 if half is not None else sg * (cur - entry) / R
                            break
                        if half is None and half_t and half_t[0] <= bt:
                            r = sg * (half_t[1] - entry) / R
                            if r <= 0:
                                res = r
                                break
                            half, cur = r, entry
                    if res is None:
                        r = sg * (c[min(j, len(c) - 1)] - entry) / R
                        res = (half + r) / 2 if half is not None else r
                    v = {tt: (float(feats[tt][0][n]), float(feats[tt][1][n])) for tt in feats}
                    kq = np.searchsorted(c15, t.to_datetime64(), side="right") - 1
                    atr15 = float(a15all.iloc[kq]) if kq >= 14 else 0.0
                    fz15, fw15, fl15 = fvg_flags(fv["15m"], t.to_datetime64(), side, entry, atr15, np.timedelta64(5, "h"))
                    fz1h, fw1h, fl1h = fvg_flags(fv["1H"], t.to_datetime64(), side, entry, atr15, np.timedelta64(24, "h"))
                    trades.append(dict(sym=sym, tf=tf, side=side, t=str(loc), stopv=stopv,
                                       R=round(res - cst / R, 3), h1=v["1H"], m15=v["15m"], h4=v["4H"],
                                       fz15=fz15, fw15=fw15, fz1h=fz1h, fw1h=fw1h, fl15=fl15, fl1h=fl1h))
        print(f"{sym}: угод {len([x for x in trades if x['sym'] == sym])}")

    def with_us(d, side):
        return d >= 1 if side == "LONG" else d <= -1
    trades.sort(key=lambda x: x["t"])
    def mark_pause(tt):
        out, cnt = [], {}
        for x in tt:
            k = (x["sym"], x["side"], x["t"][:10])
            if cnt.get(k, 0) >= 2:
                continue
            out.append(x)
            if x["R"] < 0:
                cnt[k] = cnt.get(k, 0) + 1
        return out
    filters = {
        "усі (як зараз)": lambda x: True,
        "H1 іде в бік угоди": lambda x: with_us(x["h1"][1], x["side"]),
        "без закріпу проти на M15": lambda x: not x["fz15"],
        "без закріпу проти на H1": lambda x: not x["fz1h"],
        "без закріпу проти M15 і H1": lambda x: not x["fz15"] and not x["fz1h"],
        "без FVG-стіни поруч (M15/H1)": lambda x: not x["fw15"] and not x["fw1h"],
        "без закріпу і без стіни": lambda x: not (x["fz15"] or x["fz1h"] or x["fw15"] or x["fw1h"]),
        "останній закріп M15 не проти": lambda x: not x["fl15"],
        "останній закріп H1 не проти": lambda x: not x["fl1h"],
        "останній закріп M15 і H1 не проти": lambda x: not x["fl15"] and not x["fl1h"],
        "ост. закріп M15/H1 не проти + без стіни": lambda x: not (x["fl15"] or x["fl1h"] or x["fw15"] or x["fw1h"]),
    }
    def st(rs):
        a = np.array(rs)
        if not len(a):
            return {"n": 0}
        pos, neg = a[a > 0].sum(), -a[a < 0].sum()
        eq = np.r_[0, np.cumsum(a)]
        return {"n": len(a), "win%": round(float((a > 0).mean() * 100)), "sumR": round(float(a.sum()), 1),
                "avgR": round(float(a.mean()), 2), "PF": round(float(pos / neg), 2) if neg else None,
                "DD": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    rep = {}
    mid = str((now - pd.Timedelta(days=DAYS / 2)).tz_convert(bot.LOCAL_TZ))
    for sv in ("zone", "atr1.5"):
        for name, f in list(filters.items()) + [("пауза після 2 стопів", None), ("ост. закріп M15/H1 не проти + пауза", "combo")]:
            if f == "combo":
                tt = mark_pause([x for x in trades if x["stopv"] == sv and not (x["fl15"] or x["fl1h"])])
            else:
                tt = [x for x in trades if x["stopv"] == sv and (f is None or f(x))]
            if f is None:
                tt = mark_pause(tt)
            rep[f"{sv} | {name}"] = {"all": st([x["R"] for x in tt]),
                                     **{tf: st([x["R"] for x in tt if x["tf"] == tf]) for tf in ("1m", "5m", "15m")},
                                     "5-15m 1st": st([x["R"] for x in tt if x["tf"] != "1m" and x["t"] < mid]),
                                     "5-15m 2nd": st([x["R"] for x in tt if x["tf"] != "1m" and x["t"] >= mid])}
    for k, v in rep.items():
        print("BT2", k, json.dumps(v, ensure_ascii=False))
    today = [x for x in trades if x["t"][:10] == str(now.tz_convert(bot.LOCAL_TZ).date()) and x["stopv"] == "atr1.5"]
    for x in today:
        print("TODAY", json.dumps(x, ensure_ascii=False))
