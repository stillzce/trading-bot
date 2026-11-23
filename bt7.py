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


# ------------------------------------------------------------------ bt8: варіанти трейлінгу / захисту прибутку
VARIANTS = {
    "V0_зараз": {},
    "V1_3R_трейл3R": {"A": 3, "D": 3},            # твоя ідея: 3R → БУ, 4R → +1R, 5R → +2R …
    "V2_2R_трейл2R": {"A": 2, "D": 2},
    "V3_БУ1.5R+трейл3R": {"BE": 1.5, "A": 3, "D": 3},
    "V4_трейл2ATR_від2R": {"ATR_A": 2, "ATR_K": 2},
    "V5_½на3R": {"TP": 3},
    "V6_½на2R+трейл2R": {"TP": 2, "A": 2, "D": 2},
}


def trail_variants(h, l, c, i, jend, long_, entry, stop0, R, jh, jx, px, atr_tf):
    """R кожного варіанту (без комісії). Порядок у свічці: стоп → ціль ½ → події на закритті."""
    sg = 1 if long_ else -1
    out = []
    for name, p in VARIANTS.items():
        stop = stop0
        half = None          # R першої половини
        peak = 0.0           # найкращий хід у R
        res = None
        for j in range(i + 1, jend + 1):
            lo_r = sg * ((l[j] if long_ else h[j]) - entry) / R   # найгірша точка свічки в R
            hi_r = sg * ((h[j] if long_ else l[j]) - entry) / R   # найкраща
            st_r = sg * (stop - entry) / R
            if lo_r <= st_r:                                     # стоп (за рівнем стопу)
                rest = st_r
                res = rest if half is None else (half + rest) / 2
                break
            if half is None and p.get("TP") and hi_r >= p["TP"]:  # ½ на фіксованій цілі
                half = float(p["TP"])
                stop = entry if sg * (stop - entry) < 0 else stop
            cr = sg * (c[j] - entry) / R
            if half is None and jh is not None and j == jh:       # ½ на протилежному WT свого ТФ
                if cr <= 0:
                    res = cr
                    break
                half = cr
                if sg * (stop - entry) < 0:
                    stop = entry
            if jx is not None and j == jx:                        # решта — на протилежному WT старшого ТФ
                rest = sg * (px - entry) / R
                res = rest if half is None else (half + rest) / 2
                break
            peak = max(peak, hi_r)
            new_r = None
            if p.get("BE") and peak >= p["BE"]:
                new_r = 0.0
            if p.get("A") and peak >= p["A"]:
                v = float(np.floor(peak)) - p["D"]
                new_r = v if new_r is None else max(new_r, v)
            if p.get("ATR_A") and peak >= p["ATR_A"] and atr_tf > 0:
                best = entry + sg * peak * R
                v = sg * (best - sg * p["ATR_K"] * atr_tf - entry) / R
                new_r = v if new_r is None else max(new_r, v)
            if new_r is not None and new_r > sg * (stop - entry) / R:
                stop = entry + sg * new_r * R
        if res is None:
            rest = sg * (c[jend] - entry) / R
            res = rest if half is None else (half + rest) / 2
        out.append(round(res, 3))
    return out


def report8(rows):
    rows.sort(key=lambda r: r[5])
    names = list(VARIANTS)
    sel = [r for r in rows if r[2] == "FVG" and r[4] == "with" and r[8].get(0.0) and len(r[8][0.0]) > 4]
    print("ROWS8", len(sel), "з", sel[0][5] if sel else "-", "по", sel[-1][5] if sel else "-")
    groups = {"M1–M3": ("1m", "2m", "3m"), "M5–M15": ("5m", "15m"), "M30–H4": ("30m", "1H", "2H", "4H"),
              "усі": ("1m", "2m", "3m", "5m", "15m", "30m", "1H", "2H", "4H")}
    for gname, tfs in groups.items():
        rr = [r for r in sel if r[1] in tfs]
        for k, name in enumerate(names):
            v = [r[8][0.0][4 + k] for r in rr]
            vf = [r[8][0.0][4 + k] - FEE * r[11] / r[8][0.0][3] for r in rr]
            a = np.asarray(v, float)
            print("R8", json.dumps({"g": gname, "v": name, "noFee": _st(v), "fee": _st(vf),
                                    "avgWin": round(float(a[a > 0].mean()), 2) if (a > 0).any() else 0,
                                    "big5R": int((a >= 5).sum())}, ensure_ascii=False))
    # по ТФ — з комісією
    for tf in ("1m", "2m", "3m", "5m", "15m", "30m", "1H", "2H", "4H"):
        rr = [r for r in sel if r[1] == tf]
        line = {"tf": tf}
        for k, name in enumerate(names):
            st = _st([r[8][0.0][4 + k] - FEE * r[11] / r[8][0.0][3] for r in rr])
            line[name] = [st["sumR"], st["PF"]] if st else None
        print("R8TF", json.dumps(line, ensure_ascii=False))
