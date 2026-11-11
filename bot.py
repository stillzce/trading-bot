#!/usr/bin/env python3
"""
Telegram-бот: денний контекст (PDH/PDL close vs sweep) + сигнали WaveTrend
на екстремумах (дно / вершина) у напрямку контексту, таймфрейми 1m … 1D.
Дані — TradeLocker API (окремий акаунт, НЕ проп).

Режими:
  python bot.py loop      — основний: працює в робочі години (08:00–23:00 Київ), щохвилини перевіряє сигнали
  python bot.py morning   — надіслати ранковий контекст зараз
  python bot.py once      — один прохід перевірки сигналів (для тесту)
  python bot.py symbols   — надіслати в Telegram список назв інструментів у брокера
  python bot.py test      — тестове повідомлення

Налаштування — змінні середовища (див. README.md).
"""
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================================ налаштування

def env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def need(name):
    v = env(name)
    if not v:
        sys.exit(f"Не задано змінну {name} (GitHub → Settings → Secrets and variables → Actions)")
    return v


def split(v):
    return [s.strip() for s in v.split(",") if s.strip()]


SYMBOLS = split(env("SYMBOLS", "US500,NAS100,GER40,EURUSD,GBPUSD,XAUUSD"))
SIGNAL_TFS = split(env("SIGNAL_TFS", "1m,5m,15m,30m,1H,4H,1D"))
WT_N1 = int(env("WT_CHANNEL", "10"))
WT_N2 = int(env("WT_AVERAGE", "21"))
WT_OB = float(env("WT_OB", "60"))
WT_OS = float(env("WT_OS", "-60"))
WT_SIGNAL = env("WT_SIGNAL", "exit")  # exit — вихід лінії WT із зони (як у скрипті); cross — перетин WT1/WT2 у зоні
LOCAL_TZ = ZoneInfo(env("LOCAL_TZ", "Europe/Kyiv"))
WORK_START = env("WORK_START", "08:00")
WORK_END = env("WORK_END", "23:00")
RUN_MINUTES = int(env("RUN_MINUTES", "310"))      # скільки максимум живе один запуск (ліміт GitHub — 6 год)
SESSION_TZ = ZoneInfo(env("SESSION_TZ", "America/New_York"))
SESSION_CLOSE_HOUR = int(env("SESSION_CLOSE_HOUR", "17"))  # денна свічка закривається о 17:00 NY
CONTEXT_MAX_BACK = int(env("CONTEXT_MAX_BACK", "5"))       # скільки днів назад шукати не-нейтральний контекст
DRY_RUN = env("DRY_RUN", "0") == "1"
STATE_FILE = Path(env("STATE_FILE", "state/state.json"))

TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240, "1D": 1440}
TF_LABEL = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", "1H": "H1", "4H": "H4", "1D": "D1"}
FIRST_FETCH_DAYS = {"1m": 2, "5m": 4, "15m": 8, "30m": 12, "1H": 20, "4H": 70}
HISTORY_H1_DAYS = 300   # для денних свічок і WT на D
BIAS_ICON = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "⚪"}
DAYS_UA = ["пн", "вт", "ср", "чт", "пт", "сб", "нд"]

for tf in SIGNAL_TFS:
    if tf not in TF_MIN:
        sys.exit(f"Невідомий таймфрейм у SIGNAL_TFS: {tf}. Можна: {', '.join(TF_MIN)}")


# ============================================================================ дані

EMPTY = pd.DataFrame(columns=["open", "high", "low", "close"], dtype=float)


def parse_bars(j):
    if j.get("s") == "no_data":
        return EMPTY.copy()
    d = j.get("d", j)
    if isinstance(d, dict) and "barDetails" in d:
        df = pd.DataFrame(d["barDetails"])
    elif isinstance(d, dict) and "t" in d:
        df = pd.DataFrame({k: d[k] for k in ("t", "o", "h", "l", "c") if k in d})
    elif isinstance(d, list):
        df = pd.DataFrame(d)
    else:
        raise ValueError(f"Невідомий формат відповіді TradeLocker: {str(j)[:300]}")
    if df.empty:
        return EMPTY.copy()
    df = df[["t", "o", "h", "l", "c"]].copy()
    df["t"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("t").sort_index().astype(float)
    df.columns = ["open", "high", "low", "close"]
    return df[~df.index.duplicated(keep="last")]


class TradeLocker:
    def __init__(self):
        kind = env("TL_ENV", "demo").lower()
        first = "live" if kind == "live" else "demo"
        self.s = requests.Session()
        # деякі брокери тримають демо-рахунки на live.tradelocker.com — пробуємо обидва середовища
        for e in (first, "demo" if first == "live" else "live"):
            self.base = f"https://{e}.tradelocker.com/backend-api"
            if self._login(fatal=False):
                print(f"TradeLocker: вхід через {e}.tradelocker.com")
                break
        else:
            raise RuntimeError("TradeLocker: не вдалось увійти ні в demo, ні в live — перевір TL_EMAIL, TL_PASSWORD, TL_SERVER")
        accounts = self._get("/auth/jwt/all-accounts")["accounts"]
        want = env("TL_ACC_NUM")
        acc = next((a for a in accounts if want and str(a.get("accNum")) == want), accounts[0])
        self.acc_id = acc["id"]
        self.s.headers["accNum"] = str(acc["accNum"])
        self._instruments = None
        self._ids = {}

    def _login(self, fatal=True):
        r = requests.post(
            self.base + "/auth/jwt/token",
            json={"email": need("TL_EMAIL"), "password": need("TL_PASSWORD"), "server": need("TL_SERVER")},
            timeout=30,
        )
        if not r.ok or "accessToken" not in r.text:
            msg = f"TradeLocker {self.base}: не вдалось увійти ({r.status_code}): {r.text[:300]}"
            if fatal:
                raise RuntimeError(msg)
            print(msg)
            return False
        self.s.headers["Authorization"] = f"Bearer {r.json()['accessToken']}"
        return True

    def _get(self, path, **params):
        for attempt in range(4):
            r = self.s.get(self.base + path, params=params, timeout=30)
            if r.status_code == 401 and attempt == 0:   # токен прострочився
                self._login()
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + attempt * 3)
                continue
            if not r.ok:
                raise RuntimeError(f"TradeLocker {path}: {r.status_code} {r.text[:300]}")
            return r.json()
        raise RuntimeError(f"TradeLocker {path}: не відповідає (останній код {r.status_code})")

    def instruments(self):
        if self._instruments is None:
            j = self._get(f"/trade/accounts/{self.acc_id}/instruments")
            self._instruments = j.get("d", j).get("instruments", [])
        return self._instruments

    def names(self):
        return sorted({i.get("name", "") for i in self.instruments()})

    def label(self, symbol):
        return f"TradeLocker {env('TL_SERVER', '')}".strip()

    def _resolve(self, symbol):
        if symbol in self._ids:
            return self._ids[symbol]
        ins, s = self.instruments(), symbol.upper()
        hit = next((i for i in ins if i.get("name", "").upper() == s), None) or \
            next((i for i in ins if i.get("name", "").upper().startswith(s)), None)
        if not hit:
            raise KeyError(f"інструмент «{symbol}» не знайдено — запусти режим symbols і перевір назву")
        routes = hit.get("routes", [])
        route = next((r for r in routes if r.get("type") == "INFO"), routes[0] if routes else None)
        self._ids[symbol] = (hit["tradableInstrumentId"], route["id"])
        return self._ids[symbol]

    def bars(self, symbol, resolution, frm, to=None):
        ins_id, route_id = self._resolve(symbol)
        to = to or pd.Timestamp.now(tz="UTC")
        time.sleep(0.25)  # м'яко до лімітів API
        j = self._get("/trade/history", routeId=route_id, tradableInstrumentId=ins_id, resolution=resolution,
                      **{"from": int(frm.timestamp() * 1000), "to": int(to.timestamp() * 1000)})
        return parse_bars(j)


class MockSource:
    """Фейкові дані (випадкове блукання) для локальної перевірки (DATA_SOURCE=mock)."""

    _m1 = {}

    def names(self):
        return SYMBOLS

    def _minutes(self, symbol):
        if symbol not in self._m1:
            idx = pd.date_range(pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(days=HISTORY_H1_DAYS + 5),
                                pd.Timestamp.now(tz="UTC").floor("min"), freq="1min")
            idx = idx[idx.dayofweek < 5]
            rng = np.random.default_rng(sum(map(ord, symbol)))
            base = 20000.0 if "NAS" in symbol else 1.1 if "USD" in symbol else 6000.0
            c = base * np.exp(np.cumsum(rng.normal(0, 0.0004, len(idx))))
            self._m1[symbol] = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "close": c}, index=idx)
        return self._m1[symbol]

    def bars(self, symbol, resolution, frm, to=None):
        m = self._minutes(symbol)
        m = m[(m.index >= frm) & (m.index <= (to or pd.Timestamp.now(tz="UTC")))]
        g = m.resample(f"{TF_MIN[resolution]}min", origin="epoch")
        df = pd.DataFrame({"open": g["open"].first(), "high": g["close"].max(), "low": g["close"].min(),
                           "close": g["close"].last()}).dropna()
        df["high"] = df[["high", "open"]].max(axis=1)
        df["low"] = df[["low", "open"]].min(axis=1)
        return df


class OandaSource:
    """OANDA v20 API (демо/practice): CFD-котирування без затримки. Потрібен OANDA_TOKEN."""

    note = "OANDA"
    INSTR = {"US500": "SPX500_USD", "SPX500": "SPX500_USD", "NAS100": "NAS100_USD", "GER40": "DE30_EUR",
             "DE40": "DE30_EUR", "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "XAUUSD": "XAU_USD", "US30": "US30_USD"}
    GRAN = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", "1H": "H1", "4H": "H4"}

    def __init__(self):
        live = env("OANDA_ENV", "practice").lower() == "live"
        self.base = f"https://api-{'fxtrade' if live else 'fxpractice'}.oanda.com/v3"
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {need('OANDA_TOKEN')}"
        r = self.s.get(self.base + "/accounts", timeout=30)
        if not r.ok:
            raise RuntimeError(f"OANDA: токен не підійшов ({r.status_code}): {r.text[:200]}")

    def instrument(self, symbol):
        s = symbol.upper().split(".")[0]
        return self.INSTR.get(s, symbol)

    def label(self, symbol):
        return f"OANDA {self.instrument(symbol)}"

    def names(self):
        return [f"{s} → {self.instrument(s)}" for s in SYMBOLS]

    def bars(self, symbol, resolution, frm, to=None):
        inst, gran = self.instrument(symbol), self.GRAN[resolution]
        step = pd.Timedelta(minutes=TF_MIN[resolution])
        now = pd.Timestamp.now(tz="UTC")
        rows, cur = [], frm
        for _ in range(20):  # до 20 сторінок по 5000 свічок
            r = None
            for attempt in range(3):
                r = self.s.get(f"{self.base}/instruments/{inst}/candles", timeout=30, params={
                    "granularity": gran, "price": "M", "from": cur.strftime("%Y-%m-%dT%H:%M:%SZ"), "count": 5000})
                if r.ok:
                    break
                time.sleep(2 + attempt * 3)
            if not r.ok:
                raise RuntimeError(f"OANDA {inst} {gran}: {r.status_code} {r.text[:200]}")
            candles = r.json().get("candles", [])
            for c in candles:
                m = c["mid"]
                rows.append((pd.Timestamp(c["time"]).tz_convert("UTC"), float(m["o"]), float(m["h"]),
                             float(m["l"]), float(m["c"])))
            if len(candles) < 5000 or not rows or rows[-1][0] + step >= now:
                break
            cur = rows[-1][0] + step
        if not rows:
            return EMPTY.copy()
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).set_index("t").sort_index()
        return df[~df.index.duplicated(keep="last")]


class OkxSource:
    """OKX public API (без ключів), реальний час, 24/7.
    US500 → перп на індекс S&P 500; NAS100 → перп на ETF QQQ (ціна ~в 40 разів менша, рух той самий);
    золото → XAU-USDT-SWAP. SPX-USDT-SWAP НЕ беремо: це мем-коїн SPX6900, а не індекс."""

    BASE = "https://www.okx.com/api/v5"
    CANDIDATES = {"XAUUSD": ["XAU-USDT-SWAP", "XAUT-USDT-SWAP", "XAUT-USDT"],
                  "US500": ["US500-USDT-SWAP"], "SPX500": ["US500-USDT-SWAP"],
                  "NAS100": ["QQQ-USDT-SWAP"], "NDX100": ["QQQ-USDT-SWAP"]}
    BAR = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1H": "1H", "4H": "4H"}

    def __init__(self):
        self.s = requests.Session()
        self.ids = {}
        live = set()
        for kind in ("SWAP", "SPOT"):
            r = self.s.get(f"{self.BASE}/public/instruments", params={"instType": kind}, timeout=30)
            r.raise_for_status()
            live |= {i["instId"] for i in r.json().get("data", [])}
        for sym, cands in self.CANDIDATES.items():
            hit = next((c for c in cands if c in live), None)
            if hit:
                self.ids[sym] = hit
        # для довідки: чи є на OKX інструменти на індекси
        self.extra = sorted(i for i in live if any(k in i for k in ("SPY", "QQQ", "SPX", "NAS", "NDX", "US500")))

    def has(self, symbol):
        return symbol.upper() in self.ids

    def label(self, symbol):
        inst = self.ids[symbol.upper()]
        return f"OKX {inst}" + (" (ETF QQQ, інший масштаб ціни)" if inst.startswith("QQQ") else "")

    def bars(self, symbol, resolution, frm, to=None):
        inst, bar = self.ids[symbol.upper()], self.BAR[resolution]
        frm_ms = int(frm.timestamp() * 1000)
        rows, after = [], None
        for _ in range(400):  # пагінація назад у часі по 100 свічок
            params = {"instId": inst, "bar": bar, "limit": 100}
            if after:
                params["after"] = after
            r = self.s.get(f"{self.BASE}/market/history-candles", params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                break
            rows += data
            oldest = int(data[-1][0])
            if oldest <= frm_ms or len(data) < 100:
                break
            after = oldest
            time.sleep(0.12)
        if not rows:
            return EMPTY.copy()
        df = pd.DataFrame([(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in rows],
                          columns=["t", "open", "high", "low", "close"])
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("t").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df[df.index >= frm]


class MixedSource:
    """Поки немає брокера: що є на OKX — з OKX (реальний час), решта (форекс, GER40) — з Yahoo."""

    def __init__(self):
        self.yahoo = YahooSource()
        try:
            self.okx = OkxSource()
        except Exception as e:
            print(f"OKX недоступний: {e}")
            self.okx = None
        on_okx = ", ".join(sorted(self.okx.ids)) if self.okx and self.okx.ids else ""
        self.note = f"OKX ({on_okx}) + Yahoo (решта)" if on_okx else "Yahoo"

    def pick(self, symbol):
        return self.okx if self.okx and self.okx.has(symbol) else self.yahoo

    def label(self, symbol):
        return self.pick(symbol).label(symbol)

    def min_tf(self, symbol):
        src = self.pick(symbol)
        return src.min_tf(symbol) if hasattr(src, "min_tf") else 1

    def bars(self, symbol, resolution, frm, to=None):
        return self.pick(symbol).bars(symbol, resolution, frm, to)

    def names(self):
        out = [f"{s} → {self.label(s)}" + ("" if self.min_tf(s) == 1 else ", сигнали від M15")
               for s in SYMBOLS]
        if self.okx is not None:
            out.append("OKX індекси: " + (", ".join(self.okx.extra) if self.okx.extra else "немає"))
        return out


class YahooSource:
    """Запасне джерело: Yahoo Finance (ф'ючерси/індекси/форекс), без акаунтів. Ф'ючерси CME там із затримкою ~10 хв."""

    note = "Yahoo (запасне джерело)"
    TICKERS = {"US500": "ES=F", "SPX500": "ES=F", "NAS100": "NQ=F", "NDX100": "NQ=F", "GER40": "^GDAXI",
               "DE40": "^GDAXI", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "XAUUSD": "GC=F", "US30": "YM=F"}
    INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1H": "60m", "4H": "60m"}
    MAX_DAYS = {"1m": 6.5, "5m": 58, "15m": 58, "30m": 58, "1H": 720, "4H": 720}

    def __init__(self):
        import yfinance  # noqa: F401  (перевірка, що встановлено)

    def ticker(self, symbol):
        s = symbol.upper().split(".")[0]
        return self.TICKERS.get(s, symbol)

    def label(self, symbol):
        return f"Yahoo {self.ticker(symbol)}" + ("" if self.ticker(symbol).endswith("=X") else " (затримка)")

    def min_tf(self, symbol):
        """Ф'ючерси/індекси на Yahoo із затримкою 10–30 хв — M1/M5 для них марні, лише від M15."""
        return 1 if self.ticker(symbol).endswith("=X") else 15

    def names(self):
        return [f"{s} → {self.ticker(s)}" for s in SYMBOLS]

    def bars(self, symbol, resolution, frm, to=None):
        import yfinance as yf
        now = pd.Timestamp.now(tz="UTC")
        to = to or now
        frm = max(frm, now - pd.Timedelta(days=self.MAX_DAYS[resolution]))
        for attempt in range(3):
            try:
                raw = yf.Ticker(self.ticker(symbol)).history(
                    start=frm.to_pydatetime(), end=(to + pd.Timedelta(minutes=5)).to_pydatetime(),
                    interval=self.INTERVAL[resolution], prepost=True, auto_adjust=False)
                break
            except Exception as e:
                print(f"Yahoo {symbol} {resolution}: {e}")
                time.sleep(3 + attempt * 5)
        else:
            raise RuntimeError(f"Yahoo не віддав дані для {symbol}")
        if raw is None or raw.empty:
            return EMPTY.copy()
        df = raw[["Open", "High", "Low", "Close"]].astype(float)
        df.columns = ["open", "high", "low", "close"]
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        df = df[~df.index.duplicated(keep="last")].dropna()
        if resolution == "4H":
            g = df.resample("4h", origin="epoch")
            df = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                               "low": g["low"].min(), "close": g["close"].last()}).dropna()
        return df


def make_source():
    """Пріоритет: TradeLocker → OANDA → Yahoo. Хто перший пустив — з того й беремо дані."""
    if env("DATA_SOURCE") == "mock":
        return MockSource()
    if env("DATA_SOURCE") != "yahoo":
        if env("TL_PASSWORD"):
            try:
                tl = TradeLocker()
                tl.note = "TradeLocker"
                return tl
            except Exception as e:
                print(f"{e}")
        if env("OANDA_TOKEN"):
            try:
                return OandaSource()
            except Exception as e:
                print(f"{e}")
    print("→ дані з Yahoo/OKX")
    return MixedSource()


def closed_only(df, tf):
    if df.empty:
        return df
    return df[df.index + pd.Timedelta(minutes=TF_MIN[tf]) <= pd.Timestamp.now(tz="UTC")]


class BarCache:
    """Тримає історію в пам'яті й докачує лише свіжі свічки — менше навантаження на API."""

    def __init__(self, src):
        self.src, self.data = src, {}

    def get(self, sym, tf):
        key = (sym, tf)
        now = pd.Timestamp.now(tz="UTC")
        old = self.data.get(key)
        if old is None or len(old) < 50:
            df = self.src.bars(sym, tf, now - pd.Timedelta(days=FIRST_FETCH_DAYS[tf]))
        else:
            new = self.src.bars(sym, tf, old.index[-3])
            df = pd.concat([old[old.index < new.index[0]], new]) if len(new) else old
        df = df.tail(2000)
        self.data[key] = df
        return closed_only(df, tf)


# ============================================================================ індикатори та контекст

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def wavetrend(df):
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ema(ap, WT_N1)
    d = ema((ap - esa).abs(), WT_N1)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci.fillna(0), WT_N2)
    return wt1, wt1.rolling(4).mean()


def daily_sessions(h1):
    """Денні свічки з H1, закриття о 17:00 NY. Індекс — час відкриття сесії (UTC). Лише завершені."""
    if h1.empty:
        return h1
    shift = pd.Timedelta(hours=24 - SESSION_CLOSE_HOUR)
    key = (h1.index.tz_convert(SESSION_TZ) + shift).date
    days = h1.groupby(key).agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last"))
    days = days[[d.weekday() < 5 for d in days.index]]  # без вихідних «обрізків»
    start = [pd.Timestamp(datetime.combine(d, datetime.min.time()), tz=SESSION_TZ) - shift for d in days.index]
    days.index = pd.DatetimeIndex(start).tz_convert("UTC")
    return closed_only(days, "1D")


def bias_of(y, p):
    """Контекст з однієї денної свічки y відносно попередньої p."""
    took_high, took_low = y.high > p.high, y.low < p.low
    if y.close > p.high:
        return "BULL", "закрилась вище PDH"
    if y.close < p.low:
        return "BEAR", "закрилась нижче PDL"
    if took_high and took_low:
        return "NEUTRAL", "outside-день, закрилась всередині"
    if took_high:
        return "BEAR", "зняла PDH і повернулась"
    if took_low:
        return "BULL", "зняла PDL і повернулась"
    return "NEUTRAL", "inside-день"


def day_name(ts):
    d = (ts.tz_convert(SESSION_TZ) + pd.Timedelta(hours=24 - SESSION_CLOSE_HOUR)).date()
    return f"{DAYS_UA[d.weekday()]} {d:%d.%m}"


def context(cache, sym):
    """Контекст на сьогодні; якщо вчорашня свічка нейтральна — беремо з попереднього дня."""
    now = pd.Timestamp.now(tz="UTC")
    h1 = cache.src.bars(sym, "1H", now - pd.Timedelta(days=HISTORY_H1_DAYS))
    days = daily_sessions(h1)
    if len(days) < 3:
        raise RuntimeError("замало історії для денного контексту")
    y = days.iloc[-1]
    first_bias, first_why = bias_of(y, days.iloc[-2])
    bias, why, from_day = first_bias, first_why, None
    if first_bias == "NEUTRAL":
        for k in range(2, min(CONTEXT_MAX_BACK + 1, len(days) - 1) + 1):
            b, w = bias_of(days.iloc[-k], days.iloc[-k - 1])
            if b != "NEUTRAL":
                bias, why, from_day = b, w, day_name(days.index[-k])
                break
    note = f"вчора {first_why} → контекст зі свічки {from_day}: {why}" if from_day else why
    src = cache.src.label(sym) if hasattr(cache.src, "label") else ""
    return dict(bias=bias, note=note, pdh=y.high, pdl=y.low, days=days, src=src)


def wt_extreme_cross(wt1, wt2, i):
    """Сигнал WaveTrend.
    WT_SIGNAL=exit (за замовчуванням, як у скрипті Богдана «WaveTrend BK»):
        LONG  — лінія WT виходить з дна: перетинає −60 знизу вгору (ta.crossover(tci, limitos));
        SHORT — лінія WT виходить з вершини: перетинає +60 згори вниз (ta.crossunder(tci, limitob)).
    WT_SIGNAL=cross: перетин WT1/WT2 всередині зони (старий варіант)."""
    a1, a2, b1, b2 = wt1.iloc[i - 1], wt2.iloc[i - 1], wt1.iloc[i], wt2.iloc[i]
    if WT_SIGNAL == "cross":
        if np.isnan([a1, a2, b1, b2]).any():
            return None
        if a1 <= a2 and b1 > b2 and b2 <= WT_OS:
            return "LONG"
        if a1 >= a2 and b1 < b2 and b2 >= WT_OB:
            return "SHORT"
        return None
    if np.isnan([a1, b1]).any():
        return None
    if a1 <= WT_OS < b1:
        return "LONG"
    if a1 >= WT_OB > b1:
        return "SHORT"
    return None


def zone_extreme(df, wt1, i, side, max_back=200):
    """Дно/вершина: мінімум (максимум) свічок, поки WT був у зоні перед сигналом, + свічка сигналу."""
    j = i - 1
    while j > 0 and i - j < max_back and ((wt1.iloc[j] <= WT_OS) if side == "LONG" else (wt1.iloc[j] >= WT_OB)):
        j -= 1
    seg = df.iloc[j + 1: i + 1]
    return float(seg["low"].min()) if side == "LONG" else float(seg["high"].max())


# ============================================================================ форматування і графік

def fmt(x):
    return f"{x:,.5f}".replace(",", " ") if abs(x) < 20 else f"{x:,.1f}".replace(",", " ")


def chart(df, title, pdh=None, pdl=None, mark=None, bars=90, daily=False, mark_back=0):
    wt1, wt2 = wavetrend(df)
    n = min(bars, len(df))
    df, wt1, wt2 = df.tail(n), wt1.tail(n), wt2.tail(n)
    bg, fg, grid = "#131722", "#d1d4dc", "#2a2e39"
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.04), facecolor=bg)
    x = np.arange(n)
    col = np.where((df["close"] >= df["open"]).to_numpy(), "#26a69a", "#ef5350")
    body_h = np.maximum(np.abs(df["close"] - df["open"]).to_numpy(), (df["high"].max() - df["low"].min()) * 0.0015)
    ax.vlines(x, df["low"], df["high"], color=col, lw=0.8)
    ax.bar(x, body_h, bottom=np.minimum(df["open"], df["close"]).to_numpy(), color=col, width=0.65)
    lo, hi = df["low"].min(), df["high"].max()
    span = hi - lo
    for lvl, name, c in ((pdh, "PDH", "#26a69a"), (pdl, "PDL", "#ef5350")):
        if lvl is not None and not (lo - span <= lvl <= hi + span):
            # рівень далеко від видимих свічок — не стискати графік, лише підписати
            ax.text(0.01, 0.97 if lvl > hi else 0.03, f"{name} {fmt(lvl)} {'↑' if lvl > hi else '↓'}",
                    transform=ax.transAxes, color=c, fontsize=8, va="top" if lvl > hi else "bottom")
            continue
        if lvl is not None:
            ax.axhline(lvl, color=c, ls="--", lw=1)
            ax.text(n + 13, lvl, f"{name} {fmt(lvl)}", color=c, ha="right", va="bottom", fontsize=8)
    if mark:
        i = n - 1 - mark_back  # mark_back > 0 — стрілка не на останній свічці (для прикладів)
        long_ = mark == "LONG"
        ax.annotate("▲" if long_ else "▼", (i, df["low"].iloc[i] if long_ else df["high"].iloc[i]),
                    color="#26a69a" if long_ else "#ef5350", ha="center", va="top" if long_ else "bottom", fontsize=14)
    ax2.plot(x, wt1, color="#26a69a", lw=1)
    ax2.plot(x, wt2, color="#ef5350", lw=1, ls=":")
    for lvl in (WT_OB, WT_OS):
        ax2.axhline(lvl, color="#787b86", lw=0.6, ls="--")
    ax2.axhline(0, color="#787b86", lw=0.4)
    for a in (ax, ax2):
        a.set_facecolor(bg)
        a.tick_params(colors=fg, labelsize=8)
        a.grid(color=grid, lw=0.5)
        for sp in a.spines.values():
            sp.set_color(grid)
        a.yaxis.tick_right()
    ticks = np.linspace(0, n - 1, 6).astype(int)
    ax2.set_xticks(ticks)
    if daily:
        labels = [day_name(df.index[i]) for i in ticks]
    else:
        labels = [df.index[i].tz_convert(LOCAL_TZ).strftime("%d.%m %H:%M") for i in ticks]
    ax2.set_xticklabels(labels)
    ax.set_xlim(-1, n + 14)
    ax.set_title(title, color=fg, fontsize=11, loc="left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    return buf.getvalue()


# ============================================================================ Telegram

def tg(method, data, files=None):
    if DRY_RUN:
        out = Path("out")
        out.mkdir(exist_ok=True)
        print(f"[DRY_RUN] {method}: {data.get('text') or data.get('caption') or str(data.get('media', ''))[:150]}")
        for k, (name, content) in (files or {}).items():
            (out / f"{time.time_ns()}_{k}_{name}").write_bytes(content)
        return
    token = need("TG_TOKEN")
    data = {"chat_id": need("TG_CHAT_ID"), **data}
    for attempt in range(4):
        r = requests.post(f"https://api.telegram.org/bot{token}/{method}", data=data, files=files, timeout=60)
        if r.ok:
            return
        if r.status_code == 429:
            time.sleep(r.json().get("parameters", {}).get("retry_after", 5) + 1)
            continue
        raise RuntimeError(f"Telegram {method}: {r.status_code} {r.text[:300]}")


def send_text(text):
    for i in range(0, len(text), 4000):
        tg("sendMessage", {"text": text[i:i + 4000], "parse_mode": "HTML", "disable_web_page_preview": "true"})


def send_photo(png, caption):
    tg("sendPhoto", {"caption": caption[:1024], "parse_mode": "HTML"}, files={"photo": ("chart.png", png)})


def send_album(items):
    for s in range(0, len(items), 10):
        chunk = items[s:s + 10]
        if len(chunk) == 1:
            send_photo(*chunk[0])
            continue
        files, media = {}, []
        for k, (png, cap) in enumerate(chunk):
            files[f"p{k}"] = (f"p{k}.png", png)
            media.append({"type": "photo", "media": f"attach://p{k}", "caption": cap[:1024], "parse_mode": "HTML"})
        tg("sendMediaGroup", {"media": json.dumps(media, ensure_ascii=False)}, files)


# ============================================================================ стан

def load_state():
    try:
        s = json.loads(STATE_FILE.read_text())
        return s if isinstance(s, dict) else {}
    except Exception:
        return {}


def save_state(state):
    state["sent"] = state.get("sent", [])[-3000:]
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


# ============================================================================ логіка

def build_contexts(cache):
    ctx, errors = {}, {}
    for sym in SYMBOLS:
        try:
            ctx[sym] = context(cache, sym)
        except Exception as e:
            errors[sym] = str(e)
            print(f"{sym}: контекст недоступний: {e}")
    return ctx, errors


def send_morning(ctx, errors, source_note=""):
    now = datetime.now(LOCAL_TZ)
    lines = [f"☀️ <b>Денний контекст</b> · {DAYS_UA[now.weekday()]} {now:%d.%m}"]
    if source_note and source_note != "TradeLocker":
        lines.append(f"<i>дані: {source_note}</i>")
    album = []
    for sym in SYMBOLS:
        if sym in errors:
            lines.append(f"\n⚠️ <b>{sym}</b>: {errors[sym]}")
            continue
        c = ctx[sym]
        w1, _ = wavetrend(c["days"])
        wt = w1.iloc[-1]
        zone = " (дно)" if wt <= WT_OS else " (вершина)" if wt >= WT_OB else ""
        lines.append(
            f"\n{BIAS_ICON[c['bias']]} <b>{sym}</b> — <b>{c['bias']}</b>: {c['note']}"
            f"\n   PDH {fmt(c['pdh'])} · PDL {fmt(c['pdl'])} · WT D {wt:.0f}{'↑' if wt > w1.iloc[-2] else '↓'}{zone}"
            + (f"\n   <i>дані: {c['src']}</i>" if c.get("src") else "")
        )
        png = chart(c["days"], f"{sym} · D1 · контекст {c['bias']}", c["pdh"], c["pdl"], bars=40, daily=True)
        album.append((png, f"{BIAS_ICON[c['bias']]} <b>{sym}</b> — {c['bias']}: {c['note']}"))
    send_text("\n".join(lines))
    if album:
        send_album(album)


def due(tf, minute_idx, first):
    """Чи варто цієї хвилини перевіряти TF: на закритті свічки і ще раз через хвилину (якщо брокер запізнився)."""
    if first:
        return True
    if tf == "1D":
        return False  # денна свічка закривається о 00:00 Київ — перевіряється на старті
    return minute_idx % TF_MIN[tf] in (0, 1)


def check_cycle(cache, ctx, state, first=False):
    now = pd.Timestamp.now(tz="UTC")
    minute_idx = int(now.timestamp() // 60)
    sent = state.setdefault("sent", [])
    for sym, c in ctx.items():
        hits = []  # (tf, side, df, wt1, wt2, i)
        min_tf = cache.src.min_tf(sym) if hasattr(cache.src, "min_tf") else 1
        for tf in SIGNAL_TFS:
            if not due(tf, minute_idx, first) or TF_MIN[tf] < min_tf:
                continue
            try:
                df = c["days"] if tf == "1D" else cache.get(sym, tf)
                if len(df) < 60:
                    continue
                wt1, wt2 = wavetrend(df)
                dur = pd.Timedelta(minutes=TF_MIN[tf])
                for i in range(len(df) - 3, len(df)):
                    if now - (df.index[i] + dur) > dur * 2 + pd.Timedelta(minutes=10):
                        continue  # стара свічка
                    side = wt_extreme_cross(wt1, wt2, i)
                    if not side or not ((side == "LONG" and c["bias"] == "BULL") or
                                        (side == "SHORT" and c["bias"] == "BEAR")):
                        continue
                    key = f"{sym}|{tf}|{int(df.index[i].timestamp())}|{side}"
                    if key in sent:
                        continue
                    sent.append(key)
                    hits.append((tf, side, df, wt1, wt2, i))
            except Exception as e:
                print(f"{sym} {tf}: помилка: {e}")
        if not hits:
            continue
        # кілька таймфреймів одночасно → одне повідомлення, графік зі старшого
        side = hits[0][1]
        hits.sort(key=lambda h: TF_MIN[h[0]])
        tf_top, _, df_top, _, _, i_top = hits[-1]
        icon = "🟢" if side == "LONG" else "🔴"
        where = f"вийшов з дна (перетнув {WT_OS:.0f} вгору)" if side == "LONG" else f"вийшов з вершини (перетнув {WT_OB:.0f} вниз)"
        tfs = ", ".join(TF_LABEL[h[0]] for h in hits)
        ext_name = "дно" if side == "LONG" else "вершина"
        wts = "\n".join(f"   {TF_LABEL[h[0]]}: WT {h[3].iloc[h[5]]:.0f} · {ext_name} {fmt(zone_extreme(h[2], h[3], h[5], side))}"
                        for h in hits)
        price = df_top["close"].iloc[i_top]
        caption = (
            f"{icon} <b>{side} · {sym} · {tfs}</b>\n"
            f"WT {where}:\n{wts}\n"
            f"Контекст дня: {BIAS_ICON[c['bias']]} {c['bias']} — {c['note']}\n"
            f"Ціна: {fmt(price)} · {datetime.now(LOCAL_TZ):%H:%M}"
            + (f"\nДані: {c['src']}" if c.get("src") else "")
        )
        png = chart(df_top.iloc[: i_top + 1], f"{sym} · {TF_LABEL[tf_top]} · {side}", c["pdh"], c["pdl"],
                    mark=side, daily=tf_top == "1D")
        send_photo(png, caption)


def local_dt(hhmm, base):
    h, m = map(int, hhmm.split(":"))
    return base.replace(hour=h, minute=m, second=0, microsecond=0)


def run_loop(src):
    now = datetime.now(LOCAL_TZ)
    start, end = local_dt(WORK_START, now), local_dt(WORK_END, now)
    if now.weekday() >= 5 or now >= end:
        print("Поза робочим часом — виходжу.")
        return
    if now < start - timedelta(minutes=20):
        print("Зарано — виходжу.")
        return
    if now < start:
        time.sleep((start - now).total_seconds())
    deadline = min(datetime.now(LOCAL_TZ) + timedelta(minutes=RUN_MINUTES), end)

    cache = BarCache(src)
    ctx, errors = build_contexts(cache)
    state = load_state()
    today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
    if state.get("morning") != today:
        send_morning(ctx, errors, getattr(src, "note", ""))
        state["morning"] = today
        save_state(state)

    first = True
    while datetime.now(LOCAL_TZ) < deadline:
        try:
            check_cycle(cache, ctx, state, first)
        except Exception as e:
            print(f"Помилка циклу: {e}")
        save_state(state)
        first = False
        t = time.time()
        time.sleep(max(1.0, 60 - (t % 60) + 5))  # наступна хвилина + 5 с
    print(f"Завершую о {datetime.now(LOCAL_TZ):%H:%M}.")


def run_example(src, symbol, per_tf=1, after=15):
    """Показати, як виглядали б сьогоднішні сигнали по інструменту (за контекстом дня)."""
    cache = BarCache(src)
    c = context(cache, symbol)
    start = pd.Timestamp(datetime.now(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)).tz_convert("UTC")
    found, against = [], 0
    for tf in SIGNAL_TFS:
        if tf == "1D":
            continue
        df = cache.get(symbol, tf)
        if len(df) < 60:
            continue
        wt1, wt2 = wavetrend(df)
        dur = pd.Timedelta(minutes=TF_MIN[tf])
        tf_hits = []
        for i in range(1, len(df)):
            if df.index[i] < start:
                continue
            side = wt_extreme_cross(wt1, wt2, i)
            if not side:
                continue
            if (side == "LONG" and c["bias"] == "BULL") or (side == "SHORT" and c["bias"] == "BEAR"):
                tf_hits.append((df.index[i] + dur, tf, side, i, df, wt1, wt2))
            else:
                against += 1
        found += tf_hits[-per_tf:]
    head = (f"🧪 <b>Приклад: {symbol} сьогодні</b>\nКонтекст дня: {BIAS_ICON[c['bias']]} {c['bias']} — {c['note']}\n"
            f"Сигналів за контекстом (останній на кожному ТФ): {len(found)}; проти контексту відкинуто: {against}")
    send_text(head)
    for when, tf, side, i, df, wt1, wt2 in sorted(found, key=lambda h: TF_MIN[h[1]]):
        icon = "🟢" if side == "LONG" else "🔴"
        caption = (
            f"🧪 ПРИКЛАД · {icon} <b>{side} · {symbol} · {TF_LABEL[tf]}</b>\n"
            f"WT {'вийшов з дна' if side == 'LONG' else 'вийшов з вершини'}: {wt1.iloc[i]:.0f} · "
            f"{'дно' if side == 'LONG' else 'вершина'} {fmt(zone_extreme(df, wt1, i, side))}\n"
            f"Контекст дня: {BIAS_ICON[c['bias']]} {c['bias']} — {c['note']}\n"
            f"Ціна: {fmt(df['close'].iloc[i])} · свічка закрилась {when.tz_convert(LOCAL_TZ):%H:%M}"
        )
        k = min(after, len(df) - 1 - i)  # показати, що було після сигналу
        png = chart(df.iloc[: i + 1 + k], f"{symbol} · {TF_LABEL[tf]} · {side} (приклад)", c["pdh"], c["pdl"],
                    mark=side, mark_back=k)
        send_photo(png, caption)


def run_stats(src):
    """Скільки сьогодні було сигналів WT на екстремумах: за контекстом і проти, по кожному інструменту й ТФ."""
    cache = BarCache(src)
    start = pd.Timestamp(datetime.now(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)).tz_convert("UTC")
    now = datetime.now(LOCAL_TZ)
    lines = [f"📊 <b>Сигнали WT за сьогодні</b> (з 00:00 до {now:%H:%M})",
             f"рівні ±{WT_OB:.0f} · ✅ за контекстом / ❌ проти"]
    tot_with = tot_against = 0
    for sym in SYMBOLS:
        try:
            c = context(cache, sym)
        except Exception as e:
            lines.append(f"\n⚠️ <b>{sym}</b>: {e}")
            continue
        parts, valid, s_with, s_against = [], [], 0, 0
        for tf in SIGNAL_TFS:
            if tf == "1D":
                continue
            try:
                df = cache.get(sym, tf)
            except Exception as e:
                print(f"{sym} {tf}: {e}")
                continue
            if len(df) < 60:
                continue
            wt1, wt2 = wavetrend(df)
            w = a = 0
            dur = pd.Timedelta(minutes=TF_MIN[tf])
            for i in range(1, len(df)):
                if df.index[i] < start:
                    continue
                side = wt_extreme_cross(wt1, wt2, i)
                if not side:
                    continue
                if (side == "LONG" and c["bias"] == "BULL") or (side == "SHORT" and c["bias"] == "BEAR"):
                    w += 1
                    t = (df.index[i] + dur).tz_convert(LOCAL_TZ).strftime("%H:%M")
                    valid.append(f"{TF_LABEL[tf]} {t} @ {fmt(df['close'].iloc[i])}")
                else:
                    a += 1
            s_with += w
            s_against += a
            if w or a:
                parts.append(f"{TF_LABEL[tf]} {w}/{a}")
        tot_with += s_with
        tot_against += s_against
        lines.append(f"\n{BIAS_ICON[c['bias']]} <b>{sym}</b> ({c['bias']}): ✅ {s_with} · ❌ {s_against}"
                     + (f"\n   {' · '.join(parts)}" if parts else "\n   сигналів не було"))
        if valid:
            arrow = "🟢 LONG" if c["bias"] == "BULL" else "🔴 SHORT"
            lines.append(f"   ✅ {arrow} (час закриття свічки, ціна):")
            lines += [f"      • {v}" for v in sorted(valid, key=lambda v: v.split()[1])]
    lines.append(f"\n<b>Разом:</b> ✅ {tot_with} · ❌ {tot_against}")
    send_text("\n".join(lines))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    mode, _, arg = mode.partition(":")
    if mode == "test":
        send_text("✅ Бот на зв'язку")
        return
    src = make_source()
    if mode == "symbols":
        names = src.names()
        print("\n".join(names))
        send_text(f"📋 <b>Інструменти</b> · джерело: {getattr(src, 'note', '')}\n" + "\n".join(names))
    elif mode == "morning":
        send_morning(*build_contexts(BarCache(src)), getattr(src, "note", ""))
    elif mode == "backtest":
        if arg:
            os.environ["BT_DAYS"] = arg
        import backtest
        backtest.run(src)
    elif mode == "stats":
        run_stats(src)
    elif mode == "example":
        run_example(src, (arg or "GBPUSD").upper())
    elif mode == "once":
        cache = BarCache(src)
        ctx, _ = build_contexts(cache)
        state = load_state()
        check_cycle(cache, ctx, state, first=True)
        save_state(state)
    elif mode == "loop":
        run_loop(src)
    else:
        sys.exit(f"Невідомий режим: {mode}")


if __name__ == "__main__":
    main()
