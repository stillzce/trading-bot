"""
Бектест: денний контекст (PDH/PDL close vs sweep) + сигнали WaveTrend на екстремумах у бік контексту.

Правила (механічні, щоб можна було порахувати):
  • контекст дня — як у боті (свічка до 17:00 NY проти попередньої; нейтральний день → контекст з минулого);
  • вхід — закриття свічки, на якій лінія WT вийшла з дна (перетнула −60 вгору; лонг у BULL)
    або з вершини (перетнула +60 вниз; шорт у BEAR) — як у скрипті «WaveTrend BK»;
  • стоп — за дном/вершиною: мінімум (максимум) свічок, поки WT був у зоні, + свічка сигналу;
  • виходи (рахуються паралельно): тейк 1R / 2R / 3R, ціль PDH/PDL, «тримати до кінця дня»,
    а також PD і «до кінця дня» з беззбитком після +1R;
  • усе, що не закрилось, закривається по ринку в кінці торгового дня (17:00 NY);
  • якщо в одній свічці зачеплено і стоп, і тейк — вважаємо, що спершу стоп (песимістично);
  • спред/комісії не враховуються. Угоди на різних ТФ рахуються окремо (можуть перетинатися в часі).
"""
import json
import os

import numpy as np
import pandas as pd

import bot

DAYS = int(os.environ.get("BT_DAYS", "30"))
TFS = [t for t in os.environ.get("BT_TFS", "5m,15m,30m,1H,4H").split(",") if t]
VARIANTS = ["1R", "2R", "3R", "PD", "PD_BE", "EOD", "EOD_BE"]


def session_table(h1):
    """Для кожного торгового дня: початок/кінець сесії, контекст, PDH/PDL."""
    days = bot.daily_sessions(h1)
    rows = []
    for k in range(2, len(days) + 1):
        # контекст для сесії k беремо з уже закритих днів (k-1 проти k-2, з відкатом назад)
        hist = days.iloc[:k]
        bias, why = bot.bias_of(hist.iloc[-1], hist.iloc[-2])
        if bias == "NEUTRAL":
            for b in range(2, min(bot.CONTEXT_MAX_BACK + 1, len(hist) - 1) + 1):
                bb, ww = bot.bias_of(hist.iloc[-b], hist.iloc[-b - 1])
                if bb != "NEUTRAL":
                    bias, why = bb, ww
                    break
        start = hist.index[-1] + pd.Timedelta(hours=24)
        rows.append(dict(start=start, end=start + pd.Timedelta(hours=24), bias=bias,
                         pdh=hist.iloc[-1].high, pdl=hist.iloc[-1].low))
    return rows


def simulate(df, i, side, end, pdh, pdl, wt1):
    """Повертає R для кожного варіанту виходу (або None, якщо стоп некоректний)."""
    entry = df["close"].iloc[i]
    ext = bot.zone_extreme(df, wt1, i, side)
    risk = entry - ext if side == "LONG" else ext - entry
    if not risk or risk <= 0:
        return None
    sgn = 1 if side == "LONG" else -1
    stop = entry - sgn * risk
    pd_lvl = pdh if side == "LONG" else pdl
    pd_r = (pd_lvl - entry) * sgn / risk
    pd_ok = pd_r > 0.3
    targets = {"1R": 1.0, "2R": 2.0, "3R": 3.0, "PD": pd_r if pd_ok else None, "PD_BE": pd_r if pd_ok else None,
               "EOD": None, "EOD_BE": None}
    res = {}
    future = df.iloc[i + 1:]
    future = future[future.index < end]
    for name, tgt_r in targets.items():
        if name.startswith("PD") and tgt_r is None:
            res[name] = None  # ціль уже позаду або надто близько — угоди в цьому варіанті немає
            continue
        out = None
        cur_stop, be = stop, name.endswith("_BE")
        for _, bar in future.iterrows():
            hit_stop = bar.low <= cur_stop if side == "LONG" else bar.high >= cur_stop
            hit_tgt = tgt_r is not None and ((bar.high >= entry + tgt_r * risk) if side == "LONG"
                                             else (bar.low <= entry - tgt_r * risk))
            if hit_stop:
                out = (cur_stop - entry) * sgn / risk
                break
            if hit_tgt:
                out = tgt_r
                break
            if be and ((bar.high >= entry + risk) if side == "LONG" else (bar.low <= entry - risk)):
                cur_stop = entry  # беззбиток після +1R (з наступної свічки)
        if out is None:  # закриття в кінці дня
            last = future["close"].iloc[-1] if len(future) else entry
            out = (last - entry) * sgn / risk
        res[name] = round(float(out), 3)
    res["risk_pct"] = risk / entry * 100
    return res


def stats(rs):
    rs = [r for r in rs if r is not None]
    if not rs:
        return dict(n=0)
    a = np.array(rs)
    pos, neg = a[a > 0].sum(), -a[a < 0].sum()
    eq = np.cumsum(a)
    dd = float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())
    return dict(n=len(a), win=round(float((a > 0).mean() * 100), 1), avgR=round(float(a.mean()), 2),
                sumR=round(float(a.sum()), 1), PF=round(float(pos / neg), 2) if neg > 0 else None,
                maxDD_R=round(dd, 1))


def run(src):
    now = pd.Timestamp.now(tz="UTC")
    since = now - pd.Timedelta(days=DAYS)
    trades = []
    for sym in bot.SYMBOLS:
        try:
            h1 = src.bars(sym, "1H", now - pd.Timedelta(days=DAYS + 60))
            sessions = [s for s in session_table(h1) if s["end"] > since]
        except Exception as e:
            print(f"{sym}: немає даних для контексту: {e}")
            continue
        for tf in TFS:
            try:
                df = src.bars(sym, tf, since - pd.Timedelta(days=12 if tf != "4H" else 60))
                df = bot.closed_only(df, tf)
            except Exception as e:
                print(f"{sym} {tf}: {e}")
                continue
            if len(df) < 100:
                print(f"{sym} {tf}: замало свічок ({len(df)})")
                continue
            wt1, wt2 = bot.wavetrend(df)
            dur = pd.Timedelta(minutes=bot.TF_MIN[tf])
            for s in sessions:
                if s["bias"] == "NEUTRAL":
                    continue
                idx = np.where((df.index >= s["start"]) & (df.index + dur <= s["end"]) & (df.index >= since))[0]
                for i in idx:
                    if i < 1:
                        continue
                    side = bot.wt_extreme_cross(wt1, wt2, i)
                    if not side or (side == "LONG") != (s["bias"] == "BULL"):
                        continue
                    r = simulate(df, i, side, s["end"], s["pdh"], s["pdl"], wt1)
                    if r:
                        trades.append(dict(sym=sym, tf=tf, side=side, t=str(df.index[i] + dur), **r))
            print(f"{sym} {tf}: свічок {len(df)}, угод поки {sum(1 for t in trades if t['sym'] == sym and t['tf'] == tf)}")
    trades.sort(key=lambda t: t["t"])
    report = {"period_days": DAYS, "signal": bot.WT_SIGNAL, "total_trades": len(trades), "by_tf": {}, "by_sym": {},
              "all": {v: stats([t[v] for t in trades]) for v in VARIANTS}}
    for tf in TFS:
        tt = [t for t in trades if t["tf"] == tf]
        report["by_tf"][tf] = {v: stats([t[v] for t in tt]) for v in VARIANTS}
    for sym in bot.SYMBOLS:
        tt = [t for t in trades if t["sym"] == sym]
        report["by_sym"][sym] = {v: stats([t[v] for t in tt]) for v in VARIANTS}
    report["median_risk_pct"] = round(float(np.median([t["risk_pct"] for t in trades])), 3) if trades else None
    print("BACKTEST_JSON " + json.dumps(report, ensure_ascii=False))
    os.makedirs("out", exist_ok=True)
    pd.DataFrame(trades).to_csv("out/backtest_trades.csv", index=False)
    return report
