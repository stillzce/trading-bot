"""
Бектест: денний контекст (PDH/PDL close vs sweep) + входи від дна/вершини WaveTrend у бік контексту.

WaveTrend — як у скрипті «WaveTrend BK»: tci = EMA(ci, 21), ci від hlc3 з EMA 10; зони ±60 по лінії tci.

Дно (для лонга; вершина — дзеркально), як у специфікації «WaveTrend + FVG»:
  • дно = мінімум свічок, поки tci у зоні перепроданості (≤ −60);
  • свіп: тінь нижче дна, але тіло закрилось вище → дно зсувається на новий мінімум;
  • інвалідація: тіло закрилось нижче дна → дно зламане, чекаємо нове.

Режими входу (порівнюємо):
  A «перетин»  — WT1 перетинає WT2 вгору, коли WT2 ≤ −60 (старий варіант бота);
  B «вихід»    — tci перетинає −60 знизу вгору (ta.crossover(tci, limitos) — як у скрипті і зараз у боті);
  C «FVG»      — після дна тіло свічки закривається вище найближчого незаповненого ведмежого FVG/гепу
                 і вище всіх тіней, що в нього заходили. Спрощено: лише на ТФ сигналу (без мульти-ТФ).
Для B і C — одна угода на одне дно.

Стоп — за дном (з урахуванням свіпів). Вхід — по закриттю свічки сигналу.
Виходи (паралельно): тейк 1R / 2R / 3R, ціль PDH/PDL, «до кінця дня», «БУ на 1R, далі до кінця дня».
Незакрите закривається в кінці торгового дня (17:00 NY). Стоп і тейк в одній свічці → стоп.
Спред+прослизання віднімаються з кожної угоди (COST); стоп коротший за 3×COST — вхід пропускаємо. Угоди на різних ТФ рахуються окремо.
Окремо рахуються перша і друга половина періоду — перевірка, чи результат стабільний у часі.
"""
import json
import os

import numpy as np
import pandas as pd

import bot

DAYS = int(os.environ.get("BT_DAYS", "90"))
TFS = [t for t in os.environ.get("BT_TFS", "5m,15m,30m,1H,4H").split(",") if t]
MODES = ["A", "B", "C"]
# витрати на угоду (спред + прослизання, вхід і вихід разом) у ціні джерела даних
COST = {"US500": 0.8, "NAS100": 0.06, "GER40": 2.0, "EURUSD": 0.00012, "GBPUSD": 0.00016, "AUDUSD": 0.00012, "NZDUSD": 0.00016, "XAUUSD": 0.45}
EXITS = ["1R", "2R", "3R", "PD", "EOD", "BE"]


def session_table(h1):
    days = bot.daily_sessions(h1)
    rows = []
    for k in range(2, len(days) + 1):
        hist = days.iloc[:k]
        bias, _ = bot.bias_of(hist.iloc[-1], hist.iloc[-2])
        if bias == "NEUTRAL":
            for b in range(2, min(bot.CONTEXT_MAX_BACK + 1, len(hist) - 1) + 1):
                bb, _ = bot.bias_of(hist.iloc[-b], hist.iloc[-b - 1])
                if bb != "NEUTRAL":
                    bias = bb
                    break
        # наступна торгова сесія після останнього закритого дня (вихідні пропускаємо; час — завжди 17:00 NY)
        shift = pd.Timedelta(hours=24 - bot.SESSION_CLOSE_HOUR)
        d = (hist.index[-1].tz_convert(bot.SESSION_TZ) + shift).date() + pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        start = (pd.Timestamp(d).tz_localize(bot.SESSION_TZ) - shift).tz_convert("UTC")
        end = (pd.Timestamp(d + pd.Timedelta(days=1)).tz_localize(bot.SESSION_TZ) - shift).tz_convert("UTC")
        rows.append((start, end, bias, hist.iloc[-1].high, hist.iloc[-1].low))
    return pd.DataFrame(rows, columns=["start", "end", "bias", "pdh", "pdl"])


def find_entries(o, h, l, c, w1, w2):
    """Входи в лонг (для шорта передаються дзеркальні ряди). Повертає [(i, mode, stop)]."""
    out = []
    os_ = bot.WT_OS
    active, dno, zone_start, used_b, used_c = False, np.nan, 0, False, False
    fvgs = []  # [низ, верх, межа закріпу, бар утворення]
    for i in range(2, len(c)):
        # --- дно: новий захід WT у зону = нове дно
        if w1[i] <= os_:
            if not active or w1[i - 1] > os_:
                active, dno, zone_start, used_b, used_c = True, l[i], i, False, False
            else:
                dno = min(dno, l[i])
        elif active:
            if c[i] < dno:          # тіло нижче дна — дно зламане
                active = False
            elif l[i] < dno:        # свіп — дно зсувається
                dno = l[i]
        if active:
            if w1[i - 1] <= w2[i - 1] and w1[i] > w2[i] and w2[i] <= os_:
                out.append((i, "A", dno))
            if not used_b and w1[i - 1] <= os_ < w1[i]:
                out.append((i, "B", dno))
                used_b = True
            if not used_c:
                # межа закріпу береться з ПОПЕРЕДНІХ тіней (поточна свічка має закритись вище неї)
                cands = [f for f in fvgs if f[3] >= zone_start - 3 and f[0] > dno]
                if cands:
                    near = min(cands, key=lambda f: f[0])
                    if c[i] > near[2] and c[i] > o[i]:
                        out.append((i, "C", dno))
                        used_c = True
        # --- FVG: заповнені прибираємо, тіні піднімають межу, нові додаємо
        fvgs = [f for f in fvgs if c[i] <= f[2]]
        for f in fvgs:
            if h[i] >= f[0]:
                f[2] = max(f[2], h[i])
        if h[i] < l[i - 2]:
            fvgs.append([h[i], l[i - 2], l[i - 2], i])
        elif h[i] < l[i - 1]:
            fvgs.append([h[i], l[i - 1], l[i - 1], i])
    return out


def simulate(h, l, c, i, stop, end_i, target_pd, cost=0.0):
    entry = c[i]
    risk = entry - stop
    if not risk > 0 or risk < 3 * cost:
        return None  # стоп коротший за 3 спреди — такий вхід не беремо
    fh, fl, fc = h[i + 1:end_i], l[i + 1:end_i], c[i + 1:end_i]
    eod_r = ((fc[-1] if len(fc) else entry) - entry) / risk
    pd_r = (target_pd - entry) / risk

    def run(tgt_r, be=False):
        s = stop
        for k in range(len(fc)):
            if fl[k] <= s:
                return (s - entry) / risk
            if tgt_r is not None and fh[k] >= entry + tgt_r * risk:
                return tgt_r
            if be and fh[k] >= entry + risk:
                s = entry
        return eod_r

    res = {"1R": run(1.0), "2R": run(2.0), "3R": run(3.0), "PD": run(pd_r) if pd_r > 0.3 else None,
           "EOD": run(None), "BE": run(None, be=True)}
    cr = cost / risk  # витрати в R
    out = {k: (round(float(v - cr), 3) if v is not None else None) for k, v in res.items()}
    out["cost_R"] = round(cr, 3)
    out["risk_pct"] = risk / abs(entry) * 100
    return out


def stats(rs):
    rs = [r for r in rs if r is not None]
    if not rs:
        return dict(n=0)
    a = np.array(rs)
    pos, neg = a[a > 0].sum(), -a[a < 0].sum()
    eq = np.r_[0, np.cumsum(a)]
    top3 = float(np.sort(a)[-3:].sum()) if len(a) >= 3 else float(a.sum())
    return dict(n=len(a), win=round(float((a > 0).mean() * 100), 1), avgR=round(float(a.mean()), 2),
                sumR=round(float(a.sum()), 1), PF=round(float(pos / neg), 2) if neg > 0 else None,
                maxDD_R=round(float((np.maximum.accumulate(eq) - eq).max()), 1),
                sumR_wo_top3=round(float(a.sum()) - top3, 1))


def run(src):
    now = pd.Timestamp.now(tz="UTC")
    since = now - pd.Timedelta(days=DAYS)
    trades, coverage = [], {}
    for sym in bot.SYMBOLS:
        try:
            h1 = src.bars(sym, "1H", since - pd.Timedelta(days=30))
            sess = session_table(h1)
            sess = sess[sess["end"] > since].reset_index(drop=True)
        except Exception as e:
            print(f"{sym}: немає даних для контексту: {e}")
            continue
        for tf in TFS:
            try:
                df = bot.closed_only(src.bars(sym, tf, since - pd.Timedelta(days=10 if tf != "4H" else 40)), tf)
            except Exception as e:
                print(f"{sym} {tf}: {e}")
                continue
            if len(df) < 150:
                print(f"{sym} {tf}: замало свічок ({len(df)})")
                continue
            coverage[f"{sym} {tf}"] = f"{df.index[0]:%d.%m}–{df.index[-1]:%d.%m}"
            t_close = df.index + pd.Timedelta(minutes=bot.TF_MIN[tf])
            si = np.searchsorted(sess["start"].values, df.index.values, side="right") - 1
            ends = np.searchsorted(df.index.values, sess["end"].values, side="left")
            o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
            w1, w2 = (x.to_numpy() for x in bot.wavetrend(df))
            cnt = 0
            for side in ("LONG", "SHORT"):
                if side == "LONG":
                    so, sh, sl_, sc, s1, s2 = o, h, l, c, w1, w2
                else:  # шорт = лонг на дзеркальних рядах
                    so, sh, sl_, sc, s1, s2 = -o, -l, -h, -c, -w1, -w2
                for i, mode, stop in find_entries(so, sh, sl_, sc, s1, s2):
                    k = si[i]
                    if k < 0 or k >= len(sess) or df.index[i] < since:
                        continue
                    s = sess.iloc[k]
                    if t_close[i] > s["end"] or s["bias"] != ("BULL" if side == "LONG" else "BEAR"):
                        continue
                    r = simulate(sh, sl_, sc, i, stop, int(ends[k]), s["pdh"] if side == "LONG" else -s["pdl"],
                                 COST.get(sym, 0.0))
                    if r:
                        trades.append(dict(sym=sym, tf=tf, mode=mode, side=side, t=str(t_close[i]), **r))
                        cnt += 1
            print(f"{sym} {tf}: свічок {len(df)} ({coverage[f'{sym} {tf}']}), угод {cnt}")
    trades.sort(key=lambda t: t["t"])
    rep = {"days": DAYS, "coverage": coverage, "total": len(trades), "modes": {}}
    for m in MODES:
        tm = [t for t in trades if t["mode"] == m]
        rep["modes"][m] = {
            "all": {e: stats([t[e] for t in tm]) for e in EXITS},
            "by_tf": {tf: {e: stats([t[e] for t in tm if t["tf"] == tf]) for e in EXITS} for tf in TFS},
            "by_sym": {s: {e: stats([t[e] for t in tm if t["sym"] == s]) for e in EXITS} for s in bot.SYMBOLS},
            "median_risk_pct": round(float(np.median([t["risk_pct"] for t in tm])), 3) if tm else None,
            "median_cost_R": round(float(np.median([t["cost_R"] for t in tm])), 3) if tm else None,
            "halves": {},
        }
        mid = str(since + (now - since) / 2)
        for half, cond in (("1st", lambda t: t["t"] < mid), ("2nd", lambda t: t["t"] >= mid)):
            th = [t for t in tm if cond(t)]
            rep["modes"][m]["halves"][half] = {
                "all": {e: stats([t[e] for t in th]) for e in EXITS},
                "by_tf": {tf: {e: stats([t[e] for t in th if t["tf"] == tf]) for e in EXITS} for tf in TFS},
                "by_sym": {s_: {e: stats([t[e] for t in th if t["sym"] == s_]) for e in EXITS} for s_ in bot.SYMBOLS},
            }
    print("BACKTEST_JSON " + json.dumps(rep, ensure_ascii=False))
    return rep
