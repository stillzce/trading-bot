"""Відтворення дня: прогоняє логіку бота хвилина за хвилиною на історичних даних (нічого не шле в Telegram).
python bot.py replay:08.00-10.00[:12.00]  — сигнали з 08:00 до 10:00, супровід угод до 12:00 (або до зараз)."""
import json
from datetime import datetime as _dt
import pandas as pd
import bot

SIM = {"t": None}


class FakeDT(_dt):
    @classmethod
    def now(cls, tz=None):
        t = SIM["t"].tz_convert(tz) if tz else SIM["t"].tz_convert(None)
        return _dt(t.year, t.month, t.day, t.hour, t.minute, t.second, tzinfo=t.tzinfo)


class SimSrc:
    """Джерело, що «не бачить» майбутнього: обрізає свічки до поточного часу симуляції."""
    def __init__(self, src):
        self.src, self.memo = src, {}

    def __getattr__(self, k):
        return getattr(self.src, k)

    def bars(self, sym, tf, frm, to=None):
        key = (sym, tf)
        if key not in self.memo:
            self.memo[key] = self.src.bars(sym, tf, frm)
        df = self.memo[key]
        return df[(df.index >= frm) & (df.index + pd.Timedelta(minutes=bot.TF_MIN[tf]) <= SIM["t"])]


class SimCache:
    def __init__(self, real):
        self.real, self.src, self.full = real, SimSrc(real.src), {}

    def get(self, sym, tf):
        if (sym, tf) not in self.full:
            self.full[(sym, tf)] = self.real.get(sym, tf)
        df = self.full[(sym, tf)]
        return df[df.index + pd.Timedelta(minutes=bot.TF_MIN[tf]) <= SIM["t"]]


def run(src, arg):
    day = _dt.now(bot.LOCAL_TZ).date()
    if "@" in arg:
        d, arg = arg.split("@", 1)
        day = _dt.strptime(d, "%Y-%m-%d").date()
    parts = arg.split(":")
    a, b = parts[0].split("-")
    stop_mgmt = parts[1] if len(parts) > 1 else None
    mk = lambda x: pd.Timestamp(_dt.combine(day, _dt.strptime(x, "%H.%M").time()), tz=bot.LOCAL_TZ).tz_convert("UTC")
    t0, t1 = mk(a), mk(b)
    t2 = mk(stop_mgmt) if stop_mgmt else pd.Timestamp.now(tz="UTC").floor("min")
    real = bot.BarCache(src)
    cache = SimCache(real)
    for sym in bot.SYMBOLS:  # докачати все одразу (з запасом на дні в минулому)
        for tf in ["1m", "5m", "15m", "30m", "1H", "4H"]:
            try:
                days = {"1m": 6, "5m": 8, "15m": 12, "30m": 16, "1H": 30, "4H": 80}[tf]
                df = src.bars(sym, tf, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
                cache.full[(sym, tf)] = df
            except Exception as e:
                print(f"{sym} {tf}: {e}")
    for t_ in getattr(bot, "DERIVED", {}):
        pass
    bot.DRY_RUN = True
    bot.chart = lambda *a, **k: b""
    orig_now = pd.Timestamp.now
    pd.Timestamp.now = classmethod(lambda cls, tz=None: SIM["t"] if tz is None or str(tz) == "UTC" else SIM["t"].tz_convert(tz))
    bot.datetime = FakeDT
    SIM["t"] = t0
    ctx, _ = bot.build_contexts(cache)
    COUNT = {"signals": 0, "updates": 0}
    orig_tg = bot.tg

    def tg(method, data, files=None):
        txt = data.get("text") or data.get("caption") or ""
        if method == "sendPhoto":
            COUNT["signals"] += 1
        else:
            COUNT["updates"] += 1
        print(f"\n[{SIM['t'].tz_convert(bot.LOCAL_TZ):%H:%M}] {'SIGNAL ' if method == 'sendPhoto' else ''}{txt}")
        return {"message_id": 1}
    bot.tg = tg
    state = {}
    t = t0
    first = True
    while t <= t2:
        SIM["t"] = t + pd.Timedelta(seconds=5)
        try:
            if t <= t1:
                bot.check_cycle(cache, ctx, state, first)
            bot.manage_trades(cache, ctx, state, first)
            bot.check_warnings(state)
        except Exception as e:
            print(f"помилка {t}: {e}")
        first = False
        t += pd.Timedelta(minutes=1)
    pd.Timestamp.now = orig_now
    print("\nCOUNT", COUNT)
    print("\n===== ЖУРНАЛ")
    for j in state.get("journal", []):
        print(json.dumps({k: j[k] for k in ("sym", "side", "tf", "grade", "t_open", "entry", "stop0", "R", "half", "total", "how", "t_close")}, ensure_ascii=False, default=str))
    print("===== ВІДКРИТІ")
    for j in state.get("trades", []):
        print(json.dumps({k: j[k] for k in ("sym", "side", "tf", "grade", "t_open", "entry", "stop0", "R", "half")}, ensure_ascii=False, default=str))
