# app.py
"""
Trading bot with donation_log and Telegram integration (async, python-telegram-bot v20).
- Put .env next to this file (see example in conversation).
- Modes: PAPER | LIVE (use .env MODE)
- Donation: writes donations.log and creates pending donation message with button "✅ Выполнено".
Note: TEST in PAPER before LIVE.
"""

import aiohttp
import os
import sys
import time
import hmac
import hashlib
import json
import math
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

import requests
import certifi
import urllib3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# ----------------------
# Load env
# ----------------------
load_dotenv()

RUN_ID = os.getenv("RUN_ID", "UID_" + uuid.uuid4().hex[:8])

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MODE = os.getenv("MODE", "PAPER").upper()
ALLOW_LIVE = os.getenv("ALLOW_LIVE", "0") == "1"
DRY_RUN_LIVE = os.getenv("DRY_RUN_LIVE", "1") == "1"
MIN_CAPITAL_FOR_LIVE = float(os.getenv("MIN_CAPITAL_FOR_LIVE", "11"))

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

CONTRACT_CODE = os.getenv("CONTRACT_CODE", "BTCUSDT")
API_HOST = os.getenv("API_HOST", "api.bybit.com")

DATA_DIR = os.getenv("DATA_DIR", ".")
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

START_CAPITAL = float(os.getenv("START_CAPITAL", "11.0"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.5"))
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
SMA_PERIOD = int(os.getenv("SMA_PERIOD", "200"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "10.0"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "50"))
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "0.1"))

# Donation settings
DONATION_ENABLED = os.getenv("DONATION_ENABLED", "1") == "1"
DONATION_PERCENT = float(os.getenv("DONATION_PERCENT", "5"))
DONATION_LOG_FILE = os.path.join(LOG_DIR, os.getenv("DONATION_LOG_FILE", "donations.log"))
DONATION_PENDING_FILE = os.path.join(LOG_DIR, os.getenv("DONATION_PENDING_FILE", "donations_pending.json"))
DONATION_NOTIFY = os.getenv("DONATION_NOTIFY", "1") == "1"

# ----------------------
# Logger (utf-8)
# ----------------------
logger = logging.getLogger("bot_" + RUN_ID)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, f"bot_{RUN_ID}.log"), encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ----------------------
# SSL certifi fallback
# ----------------------
REQUESTS_VERIFY = None
try:
    REQUESTS_VERIFY = certifi.where()
    # test an existing endpoint
    requests.get(f"https://{API_HOST}/v5/market/tickers", params={"category": "linear", "symbol": CONTRACT_CODE}, timeout=5, verify=REQUESTS_VERIFY)
    logger.info("certifi OK")
except Exception as e:
    logger.warning("certifi test failed: %s. Falling back to verify=False", e)
    REQUESTS_VERIFY = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------
# Utilities: indicators
# ----------------------
def sma(series, period):
    return pd.Series(series).rolling(period, min_periods=1).mean().to_numpy()

def rsi_np(prices, period=14):
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 1:
        return np.array([50.0] * len(prices))
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0.0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    up_avg = up
    down_avg = down
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        up_val = delta if delta > 0 else 0.0
        down_val = -delta if delta < 0 else 0.0
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else 0.0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def atr(high, low, close, period=14):
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(c) < 2:
        return np.full_like(c, np.nan)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[np.nan], tr])
    if len(tr) < period:
        return np.full_like(c, np.nan)
    atrv = np.full_like(c, np.nan)
    atrv[period - 1] = np.nanmean(tr[1:period])
    for i in range(period, len(c)):
        atrv[i] = (atrv[i - 1] * (period - 1) + tr[i]) / period
    return atrv

# ----------------------
# Bybit v5 async helpers
# ----------------------
BYBIT_BASE = "https://api.bybit.com"

def generate_signature_v5(api_secret, params):
    """Создание подписи для Bybit v5"""
    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(api_secret.encode(), param_str.encode(), hashlib.sha256).hexdigest()

async def get_balance_v5():
    """Асинхронное получение баланса USDT с Bybit v5"""
    ts = int(time.time() * 1000)
    params = {
        "accountType": "UNIFIED",
        "coin": "USDT",
        "api_key": BYBIT_API_KEY,
        "timestamp": ts
    }
    params["sign"] = generate_signature_v5(BYBIT_API_SECRET, params)
    url = f"{BYBIT_BASE}/v5/account/wallet-balance"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    usdt_balance = data["result"]["list"][0]["coin"][0]["walletBalance"]
                    return float(usdt_balance)
                return 0.0
    except Exception as e:
        logger.error("Bybit v5 balance error: %s", e)
        return 0.0

async def create_order_v5(side: str, symbol: str, qty: float):
    """Асинхронное создание рыночного ордера на Bybit v5"""
    ts = int(time.time() * 1000)
    params = {
        "category": "linear",
        "symbol": symbol,
        "side": side.upper(),  # BUY / SELL
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GTC",
        "api_key": BYBIT_API_KEY,
        "timestamp": ts
    }
    params["sign"] = generate_signature_v5(BYBIT_API_SECRET, params)
    url = f"{BYBIT_BASE}/v5/order/create"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=params) as resp:
                return await resp.json()
    except Exception as e:
        logger.error("Bybit v5 create order error: %s", e)
        return None

async def close_all_positions_v5(symbol: str):
    """Закрытие всех позиций по символу"""
    ts = int(time.time() * 1000)
    params = {
        "category": "linear",
        "symbol": symbol,
        "api_key": BYBIT_API_KEY,
        "timestamp": ts
    }
    params["sign"] = generate_signature_v5(BYBIT_API_SECRET, params)
    url = f"{BYBIT_BASE}/v5/position/list"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    return data
                
                # Close all positions
                for position in data["result"]["list"]:
                    size = float(position["size"])
                    if size > 0:
                        close_side = "Sell" if position["side"] == "Buy" else "Buy"
                        await create_order_v5(close_side, symbol, size)
                return data
    except Exception as e:
        logger.error("Bybit v5 close positions error: %s", e)
        return None

# ----------------------
# Account / paper ledger
# ----------------------
class Account:
    def __init__(self, start_balance=START_CAPITAL):
        self.balance = float(start_balance)
        self.start_balance = float(start_balance)
        self.open_positions = []
        self.trades = []
        self.daily_reset = datetime.utcnow().date()
        self.daily_start_balance = float(start_balance)
        self.trades_today = 0

    def reset_daily(self):
        today = datetime.utcnow().date()
        if today != self.daily_reset:
            self.daily_reset = today
            self.daily_start_balance = self.balance
            self.trades_today = 0

    def record_trade(self, trade):
        self.trades.append(trade)
        self.trades_today += 1

    def pnl_day_pct(self):
        base = self.daily_start_balance if self.daily_start_balance > 0 else 1.0
        return (self.balance - base) / base * 100.0

account = Account(START_CAPITAL)

# ----------------------
# Order helpers
# ----------------------
def calc_trade_size(price, stop_pct):
    risk_usd = account.balance * (RISK_PERCENT / 100.0)
    stop_usd = price * stop_pct
    if stop_usd <= 0:
        return 0.0, 0.0
    size_usd = risk_usd / stop_pct
    size_usd = max(size_usd, MIN_TRADE_USD)
    effective = size_usd * LEVERAGE
    contracts = effective / price
    return contracts, size_usd

def paper_create_order(side, price, size_contracts, margin_usd, stop_px=None, tp_px=None):
    if margin_usd > account.balance:
        logger.info("Insufficient balance for paper order.")
        return None
    account.balance -= margin_usd
    pos = {
        "side": side,
        "entry_price": price,
        "size": size_contracts,
        "margin_reserved": margin_usd,
        "stop_px": stop_px,
        "tp_px": tp_px,
        "entry_time": datetime.utcnow().isoformat(),
    }
    account.open_positions.append(pos)
    logger.info("PAPER OPEN %s @ %.6f size=%s margin=%.6f", side.upper(), price, size_contracts, margin_usd)
    return pos

def paper_close_position(pos, exit_price):
    if pos["side"] == "long":
        pnl = (exit_price - pos["entry_price"]) * pos["size"]
    else:
        pnl = (pos["entry_price"] - exit_price) * pos["size"]
    fee = abs(exit_price * pos["size"]) * 0.0006
    pnl_after_fee = pnl - fee
    account.balance += pos["margin_reserved"] + pnl_after_fee
    trade = {
        "entry_time": pos["entry_time"],
        "exit_time": datetime.utcnow().isoformat(),
        "entry_price": pos["entry_price"],
        "exit_price": exit_price,
        "side": pos["side"],
        "size": pos["size"],
        "pnl": pnl_after_fee,
    }
    account.record_trade(trade)
    logger.info("PAPER CLOSE %s @ %.6f pnl=%.6f new_bal=%.6f", pos["side"].upper(), exit_price, pnl_after_fee, account.balance)
    # donation hook
    try:
        if DONATION_ENABLED:
            handle_donation_on_profit(trade)
    except Exception as e:
        logger.exception("Donation hook failed: %s", e)
    return trade

# ----------------------
# Live helpers (safe with DRY_RUN_LIVE)
# ----------------------
def get_market_price():
    try:
        response = requests.get(f"{BYBIT_BASE}/v5/market/tickers", 
                               params={"category": "linear", "symbol": CONTRACT_CODE}, 
                               timeout=5, 
                               verify=REQUESTS_VERIFY)
        data = response.json()
        if data and data.get("retCode") == 0 and "list" in data["result"] and len(data["result"]["list"]) > 0:
            return float(data["result"]["list"][0]["lastPrice"])
        return 0.0
    except Exception as e:
        logger.error("Market price error: %s", e)
        return 0.0

async def get_balance_live():
    return await get_balance_v5()

async def live_create_order(side, size_contracts):
    logger.info("LIVE order requested side=%s size=%s DRY_RUN=%s", side, size_contracts, DRY_RUN_LIVE)
    if DRY_RUN_LIVE or MODE != "LIVE":
        # simulate
        return {"simulated": True, "side": side, "qty": size_contracts}
    try:
        return await create_order_v5(side, CONTRACT_CODE, size_contracts)
    except Exception as e:
        logger.error("Live create order error: %s", e)
        return None

async def live_close_all():
    logger.info("LIVE close all requested. DRY_RUN=%s", DRY_RUN_LIVE)
    if DRY_RUN_LIVE:
        return {"simulated": True}
    try:
        return await close_all_positions_v5(CONTRACT_CODE)
    except Exception as e:
        logger.error("Error closing live positions: %s", e)
        return None

# ----------------------
# Donation helpers
# ----------------------
def _load_pending():
    if not os.path.exists(DONATION_PENDING_FILE):
        return []
    try:
        with open(DONATION_PENDING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_pending(lst):
    with open(DONATION_PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)

def donation_log_write(entry: dict):
    # append to donations.log
    s = json.dumps(entry, ensure_ascii=False)
    with open(DONATION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")
    logger.info("Donation recorded: %s", s)

# This function is called on profitable trade close
def handle_donation_on_profit(trade: dict):
    pnl = trade.get("pnl", 0.0)
    if pnl <= 0:
        return
    if not DONATION_ENABLED:
        return
    donate_amount = round(pnl * (DONATION_PERCENT / 100.0), 8)
    if donate_amount <= 0:
        return
    pending = _load_pending()
    entry = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.utcnow().isoformat(),
        "trade": trade,
        "pnl": float(round(pnl, 8)),
        "donate": float(donate_amount),
        "notified": False,
    }
    pending.append(entry)
    _save_pending(pending)
    logger.info("Donation pending created: %s", entry)
    # send telegram notification if enabled
    if DONATION_NOTIFY and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        # create message with inline buttons
        text = (
            f"💰 Закрыта сделка {trade.get('side')} pnl={entry['pnl']:.8f} USDT\n"
            f"📌 Отложи на донат: {entry['donate']:.8f} USDT\n\n"
            "Нажми ✅ если выполнил перевод, или Отмена если хочешь отменить."
        )
        keyboard = [
            [
                InlineKeyboardButton("✅ Выполнено", callback_data=f"don_done:{entry['id']}"),
                InlineKeyboardButton("❌ Отменить", callback_data=f"don_cancel:{entry['id']}"),
            ]
        ]
        # we will send using the application object later via global app reference
        try:
            # use asyncio to schedule send (app is set later)
            asyncio.get_event_loop().create_task(send_telegram_notification(TELEGRAM_CHAT_ID, text, keyboard))
        except Exception as e:
            logger.exception("Failed to schedule donation notification: %s", e)

async def send_telegram_notification(chat_id, text, keyboard):
    try:
        if not app:
            logger.warning("Telegram app not ready for notifications")
            return
        await app.bot.send_message(chat_id=chat_id, text=text, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.exception("Failed to send telegram donation notification: %s", e)

# Handler for callback queries (donation done / cancel)
async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("don_done:") or data.startswith("don_cancel:"):
        action, entry_id = data.split(":", 1)
        pending = _load_pending()
        found = None
        for p in pending:
            if p.get("id") == entry_id:
                found = p
                break
        if not found:
            await query.edit_message_text("Запись не найдена или уже обработана.")
            return
        if action == "don_cancel":
            # remove pending, do not log
            pending = [p for p in pending if p.get("id") != entry_id]
            _save_pending(pending)
            await query.edit_message_text("Запись для доната отменена.")
            return
        # mark done: write to donations.log and remove pending
        rec = {
            "id": found["id"],
            "timestamp_done": datetime.utcnow().isoformat(),
            "orig_timestamp": found.get("timestamp"),
            "trade": found.get("trade"),
            "pnl": found.get("pnl"),
            "donate": found.get("donate"),
            "note": "user_marked_done"
        }
        donation_log_write(rec)
        pending = [p for p in pending if p.get("id") != entry_id]
        _save_pending(pending)
        await query.edit_message_text(f"Спасибо — донат {rec['donate']:.8f} USDT отмечен как выполненный.")
        return
    # unknown callback
    await query.edit_message_text("Неизвестная команда.")

# ----------------------
# Trading iteration (simple hourly) - similar logic as earlier
# ----------------------
async def trading_iteration(bot_send=None):
    account.reset_daily()
    if account.trades_today >= MAX_TRADES_PER_DAY:
        logger.warning("Max trades/day reached.")
        return
    if MODE == "LIVE" and not ALLOW_LIVE:
        logger.warning("LIVE requested but ALLOW_LIVE != 1.")
        return

    # fetch 1h klines from Bybit V5 API
    try:
        url = f"{BYBIT_BASE}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": CONTRACT_CODE,
            "interval": "60",
            "limit": 300  # Увеличено до 300 свечей
        }
        
        response = requests.get(url, params=params, timeout=12, verify=REQUESTS_VERIFY)
        
        # Check if response is valid
        if response.status_code != 200:
            logger.warning("HTTP error from Bybit: %s", response.status_code)
            return
            
        data = response.json()
        
        if not data or "result" not in data or "list" not in data["result"]:
            logger.warning("No kline data or invalid response structure")
            return
            
        kl = data["result"]["list"]
        
        # Логируем количество полученных свечей
        logger.info("Received %d klines for analysis", len(kl))
        
        closes = np.array([float(x[4]) for x in kl])  # index 4 is close price
        highs = np.array([float(x[2]) for x in kl])   # index 2 is high price
        lows = np.array([float(x[3]) for x in kl])    # index 3 is low price
        
    except Exception as e:
        logger.error("Failed fetch klines: %s", e)
        return

    # Проверяем достаточно ли данных для индикаторов
    min_bars_needed = max(RSI_PERIOD + 2, SMA_PERIOD + 2)
    if len(closes) < min_bars_needed:
        logger.warning("Not enough bars. Got %d, need %d", len(closes), min_bars_needed)
        return

    rsi_vals = rsi_np(closes, RSI_PERIOD)
    sma_vals = sma(closes, SMA_PERIOD)
    atr_vals = atr(highs, lows, closes, period=14)

    price = float(closes[-1])
    rsi = float(rsi_vals[-1])
    sma_now = float(sma_vals[-1])
    atr_now = float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else price * 0.01

    logger.info("Price=%.2f RSI=%.2f SMA=%.2f ATR=%.4f Bal=%.4f", price, rsi, sma_now, atr_now, account.balance)

    pnl_day = account.pnl_day_pct()
    if pnl_day <= -abs(DAILY_LOSS_LIMIT_PCT):
        logger.error("Daily loss exceeded %.2f%% -> stop", pnl_day)
        if MODE == "LIVE" and ALLOW_LIVE:
            try:
                await live_close_all()
            except Exception as e:
                logger.exception("Error closing live positions")
        global is_bot_active
        is_bot_active = False
        return

    # simple signals
    signal = None
    if rsi < 30 and price > sma_now:
        signal = "buy"
    elif rsi > 70 and price < sma_now:
        signal = "sell"

    if signal:
        atr_mult = 1.5
        stop_dist = max(atr_now * atr_mult, price * STOP_LOSS_PCT)
        stop_pct = stop_dist / price
        size_contracts, margin_usd = calc_trade_size(price, stop_pct)
        if margin_usd < MIN_TRADE_USD or size_contracts <= 0:
            logger.info("Trade too small; skip.")
            return
        tp_px = price + stop_dist * 2 if signal == "buy" else price - stop_dist * 2
        stop_px = price - stop_dist if signal == "buy" else price + stop_dist
        if MODE in ("PAPER", "BACKTEST"):
            pos = paper_create_order(signal, price * (1 + 0.0005 if signal == "buy" else 1 - 0.0005), size_contracts, margin_usd, stop_px=stop_px, tp_px=tp_px)
            if pos:
                # notify
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    try:
                        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                                   text=f"📈 PAPER OPEN {signal.upper()} {pos['entry_price']:.2f} size={pos['size']:.6f} reserved={pos['margin_reserved']:.4f}")
                    except Exception:
                        pass
        elif MODE == "LIVE" and ALLOW_LIVE:
            live_bal = await get_balance_live()
            if live_bal < MIN_CAPITAL_FOR_LIVE:
                logger.warning("Live balance %.2f < MIN required", live_bal)
                return
            resp = await live_create_order(signal, size_contracts)
            logger.info("Live create resp: %s", resp)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                try:
                    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                               text=f"⚡ LIVE order attempted {signal.upper()} (simulated={DRY_RUN_LIVE}) size={size_contracts:.4f}")
                except Exception:
                    pass
    else:
        # management: check paper positions for TP/SL and close
        to_close = []
        for pos in list(account.open_positions):
            entry = pos["entry_price"]
            side = pos["side"]
            if side == "long":
                if price <= pos.get("stop_px", entry * (1 - STOP_LOSS_PCT)) or price >= pos.get("tp_px", entry * (1 + TAKE_PROFIT_PCT)):
                    to_close.append((pos, price * 0.9995))
            else:
                if price >= pos.get("stop_px", entry * (1 + STOP_LOSS_PCT)) or price <= pos.get("tp_px", entry * (1 - TAKE_PROFIT_PCT)):
                    to_close.append((pos, price * 1.0005))
        for pos, px in to_close:
            trade = paper_close_position(pos, px)
            try:
                account.open_positions.remove(pos)
            except ValueError:
                pass
            # notify
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                try:
                    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                               text=f"🟢 PAPER CLOSE {trade['side'].upper()} pnl={trade['pnl']:.6f} new_bal={account.balance:.6f}")
                except Exception:
                    pass

# ----------------------
# Telegram handlers & control
# ----------------------
is_bot_active = False
trade_task = None
app: Optional[Application] = None  # will be set in main()

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [KeyboardButton("📊 Статус"), KeyboardButton("⚡ Торговать")],
        [KeyboardButton("⏸️ Пауза"), KeyboardButton("🔄 Баланс")],
        [KeyboardButton("❌ Стоп"), KeyboardButton("📁 Лог")]
    ]
    reply = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(f"Бот запущен (режим={MODE}). RUN_ID={RUN_ID}", reply_markup=reply)

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_bot_active, trade_task
    txt = (update.message.text or "").strip()
    if txt == "📊 Статус":
        account.reset_daily()
        await update.message.reply_text(f"Режим: {MODE}\nАктивен: {is_bot_active}\nБаланс (paper): {account.balance:.6f}\nОткрыто позиций: {len(account.open_positions)}\nСделок: {len(account.trades)}")
    elif txt == "🔄 Баланс":
        live_bal = None
        if BYBIT_API_KEY and BYBIT_API_SECRET:
            try:
                live_bal = await get_balance_live()
            except Exception as e:
                logger.warning("Get balance failed: %s", e)
        if live_bal is not None:
            await update.message.reply_text(f"Live balance: {live_bal:.6f} USDT")
        else:
            await update.message.reply_text(f"Paper balance: {account.balance:.6f} USDT")
    elif txt == "⚡ Торговать":
        if not is_bot_active:
            is_bot_active = True
            await update.message.reply_text("Торговля включена.")
            trade_task = asyncio.create_task(trade_loop(context.bot))
        else:
            await update.message.reply_text("Уже запущено.")
    elif txt == "⏸️ Пауза":
        is_bot_active = False
        await update.message.reply_text("Торговля приостановлена.")
    elif txt == "❌ Стоп":
        is_bot_active = False
        # close paper positions
        current_price = get_market_price()
        for pos in list(account.open_positions):
            paper_close_position(pos, current_price)
            try:
                account.open_positions.remove(pos)
            except Exception:
                pass
        if MODE == "LIVE" and ALLOW_LIVE:
            await live_close_all()
        await update.message.reply_text("Бот остановлен и позиции закрыты.")
    elif txt == "📁 Лог":
        # send donation log and trades log if exist
        files = []
        if os.path.exists(DONATION_LOG_FILE):
            files.append(DONATION_LOG_FILE)
        trades_csv = os.path.join(LOG_DIR, "trades.csv")
        if os.path.exists(trades_csv):
            files.append(trades_csv)
        if files:
            for f in files:
                try:
                    await context.bot.send_document(chat_id=update.effective_chat.id, document=open(f, "rb"))
                except Exception as e:
                    logger.error("Failed to send file %s: %s", f, e)
        else:
            await update.message.reply_text("Логов не найдено.")
    else:
        await update.message.reply_text("Команда не распознана. Используй клавиатуру.")

async def trade_loop(bot=None):
    global is_bot_active
    logger.info("Trade loop started.")
    while is_bot_active:
        try:
            await trading_iteration(bot_send=bot)
        except Exception as e:
            logger.exception("Error in trading iteration: %s", e)
        await asyncio.sleep(60 * 60)  # hourly
    logger.info("Trade loop stopped.")

# ----------------------
# Persistence
# ----------------------
def save_trade_logs():
    try:
        with open(os.path.join(LOG_DIR, "trade_log.json"), "w", encoding="utf-8") as f:
            json.dump({"start_balance": account.start_balance, "end_balance": account.balance, "trades": account.trades}, f, ensure_ascii=False, indent=2)
        if account.trades:
            pd.DataFrame(account.trades).to_csv(os.path.join(LOG_DIR, "trades.csv"), index=False, encoding="utf-8")
        logger.info("Saved logs")
    except Exception as e:
        logger.exception("Save logs failed: %s", e)

# ----------------------
# Main
# ----------------------
def main():
    global app, account, MODE
    if not TELEGRAM_BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN in .env")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    application.add_handler(CallbackQueryHandler(callback_query_handler))

    app = application

    # initialize account
    account = Account(START_CAPITAL)

    logger.info("Bot start RUN_ID=%s MODE=%s CONTRACT=%s", RUN_ID, MODE, CONTRACT_CODE)
    
    try:
        application.run_polling()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        save_trade_logs()

if __name__ == "__main__":
    main()