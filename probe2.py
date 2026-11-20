import requests, pandas as pd, yfinance as yf
def kraken(pair):
    js = requests.get("https://api.kraken.com/0/public/OHLC", params={"pair": pair, "interval": 1}, timeout=20).json()
    key = [k for k in js["result"] if k != "last"][0]
    return pd.DataFrame([[pd.Timestamp(int(k[0]), unit="s", tz="UTC"), float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                         for k in js["result"][key]], columns=["t", "open", "high", "low", "close"]).set_index("t")
def run():
    for p, y in (("EURUSD", "EURUSD=X"), ("GBPUSD", "GBPUSD=X"), ("AUDUSD", "AUDUSD=X")):
        k = kraken(p)
        yh = yf.Ticker(y).history(period="1d", interval="1m").rename(columns=str.lower)
        yh.index = yh.index.tz_convert("UTC")
        j = k[["close"]].join(yh[["close"]], rsuffix="_y", how="inner")
        d = (j["close"] - j["close_y"]) * 10000
        print(f"{p}: спільних хвилин {len(j)} · різниця Kraken−Yahoo в пунктах: середня {d.mean():.1f}, розкид {d.std():.1f}, "
              f"кореляція змін {j['close'].diff().corr(j['close_y'].diff()):.2f}")
        if p == "EURUSD":
            w = k[(k.index >= "2026-11-20 06:45") & (k.index < "2026-11-20 07:45")]
            print("Kraken EURUSD 08:45–09:45 Київ: макс", w["high"].max(), "о", w["high"].idxmax().tz_convert("Europe/Kyiv").strftime("%H:%M"),
                  "· мін", w["low"].min(), "о", w["low"].idxmin().tz_convert("Europe/Kyiv").strftime("%H:%M"))
            for t in ("06:48", "06:50", "07:00", "07:05", "07:20", "07:30", "07:38", "07:45"):
                r = k.loc[pd.Timestamp(f"2026-11-20 {t}", tz="UTC")]
                print("  ", pd.Timestamp(f"2026-11-20 {t}", tz="UTC").tz_convert("Europe/Kyiv").strftime("%H:%M"), "H", r["high"], "L", r["low"], "C", r["close"])
            k3 = k.resample("3min").agg({"high": "max", "low": "min", "close": "last"})
            w3 = k3[(k3.index >= "2026-11-20 06:30") & (k3.index < "2026-11-20 07:50")]
            print("3m lows:", " ".join(f"{t.tz_convert('Europe/Kyiv'):%H:%M}:{v:.5f}" for t, v in w3["low"].items()))
