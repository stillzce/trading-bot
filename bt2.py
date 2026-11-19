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
                for stopv in ("zone", "atr1", "atr1.5", "atr2"):
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
                    trades.append(dict(sym=sym, tf=tf, side=side, t=str(loc), stopv=stopv,
                                       R=round(res - cst / R, 3), h1=v["1H"], m15=v["15m"], h4=v["4H"]))
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
        "H1 проти": lambda x: not with_us(x["h1"][1], x["side"]),
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
    for sv in ("zone", "atr1", "atr1.5", "atr2"):
        for name, f in list(filters.items()) + [("пауза після 2 стопів", None)]:
            tt = [x for x in trades if x["stopv"] == sv and (f is None or f(x))]
            if f is None:
                tt = mark_pause(tt)
            rep[f"{sv} | {name}"] = {"all": st([x["R"] for x in tt]),
                                     **{tf: st([x["R"] for x in tt if x["tf"] == tf]) for tf in ("1m", "5m", "15m")},
                                     "5-15m 1st": st([x["R"] for x in tt if x["tf"] != "1m" and x["t"] < mid]),
                                     "5-15m 2nd": st([x["R"] for x in tt if x["tf"] != "1m" and x["t"] >= mid])}
    for k, v in rep.items():
        print("BT2", k, json.dumps(v, ensure_ascii=False))
    today = [x for x in trades if x["t"][:10] == str(now.tz_convert(bot.LOCAL_TZ).date()) and x["stopv"] == "zone"]
    for x in today:
        print("TODAY", json.dumps(x, ensure_ascii=False))
