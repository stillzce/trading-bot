"""bt7: крипта (BTC/ETH/SOL), правила демо-бота: 24/7, доба UTC з вихідними.
Порівняння: «зараз» (вхід на виході WT у бік контексту + 🔄 закріп за FVG проти контексту з чистим шляхом ≥1.5 ATR)
проти «закріп по контексту» (у бік контексту вхід лише на закріпі тіла за FVG після дна/вершини WT; 🔄 — так само).
R рахується без комісії і з комісією OKX (тейкер 0.05% × 2 від обсягу позиції).
Запуск: Actions → okx-demo → mode bt7 (python bot.py bt7)."""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import bot

FEE = float(os.environ.get("BT_FEE", "0.001"))  # вхід + вихід


def session_table_crypto(h1):
    days = bot.daily_sessions(h1, crypto=True)
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
        start = hist.index[-1] + pd.Timedelta(days=1)
        rows.append((start, start + pd.Timedelta(days=1), bias, hist.iloc[-1].high, hist.iloc[-1].low))
    return pd.DataFrame(rows, columns=["start", "end", "bias", "pdh", "pdl"])


def _st(a):
    a = np.asarray(a, float)
    if not len(a):
        return None
    w, ls = a[a > 0], a[a <= 0]
    eq = np.r_[0, np.cumsum(a)]
    return {"n": len(a), "win%": round(float(len(w) / len(a) * 100)), "sumR": round(float(a.sum()), 1),
            "PF": round(float(w.sum() / -ls.sum()), 2) if ls.sum() < 0 else None,
            "avgR": round(float(a.mean()), 3), "DD": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}


def report7(rows):
    rows.sort(key=lambda r: r[5])
    K = 0.0
    print("ROWS", len(rows), "з", rows[0][5] if rows else "-", "по", rows[-1][5] if rows else "-")

    def counter(r):  # 🔄 проти контексту: закріп за FVG + чистий шлях ≥ 1.5 ATR
        return r[2] == "FVG" and r[4] == "against" and r[7] >= 1.5

    variants = {
        "NOW": lambda r: (r[2] == "WT" and r[4] == "with") or counter(r),
        "FVGCTX": lambda r: (r[2] == "FVG" and r[4] == "with") or counter(r),
        "NOW_ctx_only": lambda r: r[2] == "WT" and r[4] == "with",
        "FVGCTX_ctx_only": lambda r: r[2] == "FVG" and r[4] == "with",
        "counter_only": counter,
    }
    for name, f in variants.items():
        sel = [r for r in rows if f(r) and r[8].get(K)]
        g = defaultdict(lambda: defaultdict(list))
        for r in sel:
            Rk = r[8][K]
            v = Rk[1]
            fee = FEE * r[11] / Rk[3] if Rk[3] > 0 else 0
            vf = v - fee
            for grp, key in (("all", "all"), ("tf", r[1]), ("sym", r[0]), ("month", r[5][:7]), ("side", r[3]),
                             ("ctx", r[4])):
                g[grp][key].append((v, vf))
        out = {grp: {k: {"noFee": _st([x[0] for x in vals]), "fee": _st([x[1] for x in vals])}
                     for k, vals in d.items()} for grp, d in g.items()}
        print("R7", json.dumps({"v": name, **out}, ensure_ascii=False))
