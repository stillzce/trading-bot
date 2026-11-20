"""Перевірка крипто-бірж як джерела FX: які пари є, скільки історії, яка точність."""
import time
import requests
import pandas as pd

PAIRS = ["EUR", "GBP", "AUD", "NZD"]


def show(name, df):
    if df is None or df.empty:
        print(f"{name}: порожньо")
        return
    steps = df["close"].diff().abs()
    steps = steps[steps > 0]
    print(f"{name}: {len(df)} свічок {df.index[0]} → {df.index[-1]} · остання {df['close'].iloc[-1]} · "
          f"мін. крок {steps.min() if len(steps) else None} · без руху {(df['high'] == df['low']).mean() * 100:.0f}% · "
          f"обсяг/хв {df['vol'].mean():.1f}")


def get(url, **kw):
    r = requests.get(url, params=kw, timeout=20)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    return r.json()


def run():
    for p in PAIRS:
        # Binance spot XXXUSDT
        for host in ("https://api.binance.com", "https://data-api.binance.vision", "https://api.binance.us"):
            try:
                js = get(f"{host}/api/v3/klines", symbol=f"{p}USDT", interval="1m", limit=1000)
                df = pd.DataFrame([[pd.Timestamp(k[0], unit="ms", tz="UTC"), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                                   for k in js], columns=["t", "high", "low", "close", "vol"]).set_index("t")
                show(f"Binance {host.split('//')[1]} {p}USDT", df)
            except Exception as e:
                print(f"Binance {host} {p}USDT: {e}")
        # Bybit spot
        try:
            js = get("https://api.bybit.com/v5/market/kline", category="spot", symbol=f"{p}USDT", interval="1", limit=1000)
            rows = js.get("result", {}).get("list", [])
            df = pd.DataFrame([[pd.Timestamp(int(k[0]), unit="ms", tz="UTC"), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                               for k in rows], columns=["t", "high", "low", "close", "vol"]).set_index("t").sort_index()
            show(f"Bybit {p}USDT", df) if rows else print(f"Bybit {p}USDT: {js.get('retMsg')}")
        except Exception as e:
            print(f"Bybit {p}USDT: {e}")
        # Kraken fiat XXXUSD
        try:
            js = get("https://api.kraken.com/0/public/OHLC", pair=f"{p}USD", interval=1)
            if js.get("error"):
                print(f"Kraken {p}USD: {js['error']}")
            else:
                key = [k for k in js["result"] if k != "last"][0]
                df = pd.DataFrame([[pd.Timestamp(int(k[0]), unit="s", tz="UTC"), float(k[2]), float(k[3]), float(k[4]), float(k[6])]
                                   for k in js["result"][key]], columns=["t", "high", "low", "close", "vol"]).set_index("t")
                show(f"Kraken {p}USD ({key})", df)
        except Exception as e:
            print(f"Kraken {p}USD: {e}")
        # Bitstamp fiat
        try:
            js = get(f"https://www.bitstamp.net/api/v2/ohlc/{p.lower()}usd/", step=60, limit=1000)
            rows = js.get("data", {}).get("ohlc", [])
            df = pd.DataFrame([[pd.Timestamp(int(k["timestamp"]), unit="s", tz="UTC"), float(k["high"]), float(k["low"]),
                                float(k["close"]), float(k["volume"])] for k in rows],
                              columns=["t", "high", "low", "close", "vol"]).set_index("t")
            show(f"Bitstamp {p}USD", df)
        except Exception as e:
            print(f"Bitstamp {p}USD: {e}")
        # OKX
        for inst in (f"{p}-USDT", f"{p}-USD", f"{p}-USDT-SWAP"):
            try:
                js = get("https://www.okx.com/api/v5/market/candles", instId=inst, bar="1m", limit=300)
                rows = js.get("data", [])
                if not rows:
                    print(f"OKX {inst}: {js.get('msg') or 'немає'}")
                    continue
                df = pd.DataFrame([[pd.Timestamp(int(k[0]), unit="ms", tz="UTC"), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                                   for k in rows], columns=["t", "high", "low", "close", "vol"]).set_index("t").sort_index()
                show(f"OKX {inst}", df)
            except Exception as e:
                print(f"OKX {inst}: {e}")
        # Yahoo для порівняння
        try:
            import yfinance as yf
            d = yf.Ticker(f"{p}USD=X").history(period="1d", interval="1m")
            d = d.rename(columns=str.lower)
            d["vol"] = d["volume"]
            show(f"Yahoo {p}USD=X", d)
        except Exception as e:
            print(f"Yahoo {p}: {e}")
        time.sleep(1)
