"""Дослідження: які FVG «живі» (ціна реагує на першому дотику), а які ціна пролітає.
python bot.py fvgstudy — BTC/ETH/SOL, 6 місяців хвилинок OKX, ТФ від M1 до H12.

Для кожного FVG (бичачого — розрив між high[j-2] і low[j]; ведмежого — дзеркально):
  ознаки в момент утворення: розмір/ATR, тіло та діапазон імпульсної свічки, обсяг, злам структури (BOS),
  зняття ліквідності, положення в діапазоні (дисконт/премія), WT у зоні, перекриття з FVG старшого ТФ.
  результат першого дотику (до 300 свічок свого ТФ):
    «живий»  — від дотику ціна пішла від FVG на ≥ 1 ATR раніше, ніж тіло закрилося за дальньою межею;
    «пробитий» — тіло закрилося за дальньою межею раніше.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import bot
from bt4 import load, resample

TFS = ["1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "12H"]
MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "2H": 120, "4H": 240, "12H": 720}
HORIZON = 300


def fvgs_of(d, tf):
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    v = d["volume"].to_numpy() if "volume" in d.columns else np.zeros(len(d))
    n = len(d)
    atr = pd.Series(h - l).rolling(14).mean().to_numpy()
    w1 = bot.wavetrend(d)[0].to_numpy()
    tclose = (d.index + pd.Timedelta(minutes=MIN[tf])).values.astype("datetime64[m]").astype(np.int64)
    out = []
    for sg in (1, -1):
        H = h if sg == 1 else -l
        L = l if sg == 1 else -h
        C = c * sg
        O = o * sg
        W = w1 * sg
        idx = np.nonzero(L[2:] > H[:-2])[0] + 2          # бичачий у «лонгових» координатах
        for j in idx:
            if j < 60 or np.isnan(atr[j]) or atr[j] <= 0:
                continue
            bot_, top = H[j - 2], L[j]
            a = atr[j]
            m = j - 1
            rng = H[m] - L[m]
            body = abs(C[m] - O[m]) / rng if rng > 0 else 0
            vr = v[m] / np.mean(v[j - 21:j - 1]) if v[j - 21:j - 1].mean() > 0 else np.nan
            bos = C[m] > H[j - 22:j - 1].max()
            sweep = min(L[j - 2], L[m]) < L[j - 32:j - 2].min()
            lo50, hi50 = L[j - 50:j].min(), H[j - 50:j].max()
            loc = ((bot_ + top) / 2 - lo50) / (hi50 - lo50) if hi50 > lo50 else 0.5
            wt_ext = np.nanmin(W[j - 6:j + 1]) <= bot.WT_OS
            # перший дотик і результат
            res, age = None, None
            end = min(n, j + 1 + HORIZON)
            k = j + 1
            while k < end:
                if L[k] <= top:                           # ціна повернулась у FVG
                    age = k - j
                    for q in range(k, end):
                        if C[q] < bot_:
                            res = 0
                            break
                        if H[q] >= top + a:
                            res = 1
                            break
                    break
                k += 1
            if res is None:
                continue
            out.append(dict(tf=tf, sg=sg, t=int(tclose[j]), lo=float(bot_ * sg), hi=float(top * sg),
                            size=(top - bot_) / a, body=body, rng=rng / a, vr=vr, bos=bool(bos),
                            sweep=bool(sweep), loc=loc, wt=bool(wt_ext), age=age, res=res,
                            tfill=None))
    return out


def rate(rows):
    if not rows:
        return None
    r = np.array([x["res"] for x in rows])
    return {"n": len(r), "живі%": round(float(r.mean() * 100), 1)}


def run(src):
    now = pd.Timestamp.now(tz="UTC")
    okx = src if type(src).__name__ == "OkxSource" else getattr(src, "okx", None)
    allf = []
    for sym in bot.SYMBOLS:
        t0 = time.time()
        data, _ = load(sym, okx, now)
        m1 = data["1m"]
        per = {}
        for tf in TFS:
            d = m1 if tf == "1m" else resample(m1, MIN[tf])
            per[tf] = fvgs_of(d, tf)
        # перекриття з «живим» на момент утворення FVG старшого ТФ (того ж напрямку, утвореним раніше)
        arr = {}
        for tf2 in TFS:
            G = per[tf2]
            arr[tf2] = {k: np.array([g[k] for g in G], dtype=float) for k in ("sg", "t", "lo", "hi")}
            arr[tf2]["tend"] = arr[tf2]["t"] + np.array([g["age"] for g in G], dtype=float) * MIN[tf2]
        for i, tf in enumerate(TFS):
            for f in per[tf]:
                f["htf"] = False
                for tf2 in TFS[i + 1:]:
                    A = arr[tf2]
                    if not len(A["t"]):
                        continue
                    m = (A["sg"] == f["sg"]) & (A["t"] <= f["t"]) & (A["tend"] >= f["t"]) & \
                        (A["lo"] <= f["hi"]) & (f["lo"] <= A["hi"])
                    if m.any():
                        f["htf"] = True
                        break
                f["sym"] = sym
            allf += per[tf]
        print(f"{sym}: {sum(len(v) for v in per.values())} FVG, {time.time() - t0:.0f} c", flush=True)
    print("FVGALL", json.dumps({"усі": rate(allf)}, ensure_ascii=False))
    groups = {"M1–M5": ("1m", "3m", "5m"), "M15–H1": ("15m", "30m", "1H"), "H2–H12": ("2H", "4H", "12H")}
    for tf in TFS:
        print("FVGTF", json.dumps({"tf": tf, **(rate([f for f in allf if f["tf"] == tf]) or {})}, ensure_ascii=False))
    feats = {
        "розмір <0.2 ATR": lambda f: f["size"] < 0.2,
        "розмір 0.2–0.5": lambda f: 0.2 <= f["size"] < 0.5,
        "розмір 0.5–1": lambda f: 0.5 <= f["size"] < 1,
        "розмір ≥1 ATR": lambda f: f["size"] >= 1,
        "імпульс: тіло ≥70%": lambda f: f["body"] >= 0.7,
        "імпульс: тіло <40%": lambda f: f["body"] < 0.4,
        "імпульс: діапазон ≥2 ATR": lambda f: f["rng"] >= 2,
        "імпульс: діапазон <1 ATR": lambda f: f["rng"] < 1,
        "обсяг ≥2× середнього": lambda f: f["vr"] == f["vr"] and f["vr"] >= 2,
        "обсяг <1×": lambda f: f["vr"] == f["vr"] and f["vr"] < 1,
        "злам структури (BOS)": lambda f: f["bos"],
        "без BOS": lambda f: not f["bos"],
        "зняв ліквідність": lambda f: f["sweep"],
        "у дисконті (для бичачого)": lambda f: f["loc"] < 0.5,
        "у премії (для бичачого)": lambda f: f["loc"] >= 0.5,
        "WT у зоні": lambda f: f["wt"],
        "перекриття з FVG старшого ТФ": lambda f: f["htf"],
        "дотик ≤5 свічок": lambda f: f["age"] <= 5,
        "дотик 6–30 свічок": lambda f: 6 <= f["age"] <= 30,
        "дотик >30 свічок": lambda f: f["age"] > 30,
    }
    for g, tfs in groups.items():
        rows = [f for f in allf if f["tf"] in tfs]
        base = rate(rows)
        out = {"g": g, "база": base}
        for name, fn in feats.items():
            out[name] = rate([f for f in rows if fn(f)])
        print("FVGF", json.dumps(out, ensure_ascii=False))
    # найкращі комбінації (жадібно, мінімум 3% вибірки)
    for g, tfs in groups.items():
        rows = [f for f in allf if f["tf"] in tfs]
        chosen, cur = [], rows
        while True:
            best = None
            for name, fn in feats.items():
                if name in chosen:
                    continue
                sub = [f for f in cur if fn(f)]
                if len(sub) < max(200, 0.03 * len(rows)):
                    continue
                rr = np.mean([f["res"] for f in sub])
                if best is None or rr > best[1]:
                    best = (name, rr, sub)
            if not best or best[1] <= np.mean([f["res"] for f in cur]) + 0.01:
                break
            chosen.append(best[0])
            cur = best[2]
            if len(chosen) >= 4:
                break
        print("FVGC", json.dumps({"g": g, "фільтри": chosen, **(rate(cur) or {}),
                                  "база": rate(rows)}, ensure_ascii=False))
