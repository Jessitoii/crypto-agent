import asyncio
import time
import os
import json
import torch
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
from setfit import SetFitModel
import re


LOCAL_BRAIN_MODE = True
if LOCAL_BRAIN_MODE:
    from model import NexusPredictor
    local_brain = NexusPredictor("standard_deberta_nexus")
else:
    local_brain = None
# Proje Modülleri
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_CHANNELS, API_ID, API_HASH, TELETHON_SESSION_NAME, STARTING_BALANCE
from main import BotContext, SharedState
from binance_client import BinanceExecutionEngine
from services import process_news, ensure_fresh_data
from utils import find_coins, get_top_100_map
from price_buffer import PriceBuffer
from exchange import PaperExchange
from brain import AgentBrain
from config import GROQCLOUD_API_KEY, GROQCLOUD_MODEL, GOOGLE_API_KEY, GEMINI_MODEL


def clean_news_text(text):
    # 1. URL'leri temizle (http, https)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # 2. Telegram/Twitter handle'larını temizle (@cointelegraph vb.)
    text = re.sub(r'@\w+', '', text)
    # 3. Markdown link kalıntılarını ve buton metinlerini temizle
    text = re.sub(r'\[News\]\(.*?\)|\[Markets\]\(.*?\)|\[YouTube\]\(.*?\)', '', text, flags=re.IGNORECASE)
    # 4. Kalınlaştırma (**) işaretlerini kaldır
    text = text.replace('**', '')
    # 5. Gereksiz boşlukları ve satır başlarını temizle
    text = text.replace('**', '').replace('🚨 NOW:', '').replace('🚨 BREAKING:', '')
    text = text.replace("[— link]( ", "")
    # Gereksiz boşlukları al
    return " ".join(text.split()).strip()

# 1. DATABASE'İ DEVRE DIŞI BIRAKAN MOCK
class MockMemory:
    def is_duplicate(self, text): return False, 0.0
    def add_news(self, source, content): pass
    def log_decision(self, record): return 999 # Fake ID
    def log_trade(self, record, decision_id=None): pass

# [NEW] SETFIT MODEL WRAPPER

async def get_historical_technicals(ctx, pair, msg_ts):
    """Haber anındaki teknik metrikleri hesaplar."""
    # 1. Hedef Coin için geçmiş 100 dakikayı çek (RSI ve Changes için)
    # 100 dakika çekiyoruz ki RSI (14) sağlıklı hesaplansın
    klines = await ctx.real_exchange.client.futures_klines(
        symbol=pair.upper(),
        interval='1m',
        endTime=int(msg_ts * 1000),
        limit=100
    )
    
    if not klines:
        return None

    # Buffer oluştur ve doldur
    temp_buffer = PriceBuffer()
    for k in klines:
        # (price, timestamp, is_closed)
        temp_buffer.update_candle(float(k[4]), k[0]/1000, True)
    
    # Anlık fiyatı son kapanışa eşitle
    temp_buffer.current_price = float(klines[-1][4])
    
    # 2. BTC Trendi için aynı işlemi yap
    btc_klines = await ctx.real_exchange.client.futures_klines(
        symbol="BTCUSDT",
        interval='1m',
        endTime=int(msg_ts * 1000),
        limit=60
    )
    
    btc_trend = 0.0
    if btc_klines:
        btc_start = float(btc_klines[0][4])
        btc_end = float(btc_klines[-1][4])
        btc_trend = ((btc_end - btc_start) / btc_start) * 100

    return {
        'price': temp_buffer.current_price,
        'rsi': temp_buffer.calculate_rsi(),
        'changes': temp_buffer.get_all_changes(),
        'btc_trend': btc_trend,
    }

coin_map = get_top_100_map()
async def simulate_process_news(message, ctx, f_log):
    """
    services.py -> process_news() fonksiyonunun simülasyon versiyonu.
    """
    msg_text = message.text
    msg_ts = message.date.timestamp()
    msg_dt = message.date.strftime("%Y-%m-%d %H:%M:%S")

    # --- 1. FİLTRELEME (is_duplicate benzeri) ---
    is_dup, _ = ctx.memory.is_duplicate(msg_text)
    if is_dup: return

    # --- 2. COIN TESPİTİ (Regex + AI Fallback) ---
    detected_pairs = find_coins(msg_text, coin_map=coin_map)
    
    if not detected_pairs:
        # regex first
        pass
        
        # AI Fallback: Sadece regex ile bulamazsak, brain kullanabiliriz.
        # Ancak SetFit sadece classification yapıyor, entity extraction yapmıyor.
        # Bu yüzden burada regex ile bulunamayanı atlayacağız veya eski brain'i entity extraction için tutacağız.
        # Kullanıcı "deberta modelime uygun olacak şekilde" dediği için, LLM yerine Regex+SetFit odaklanıyoruz.
        # Ancak kodda 'brain' duruyor, eğer symbol bulunamazsa eski brain'i (detect_symbol) kullanabiliriz.
        found_symbol = await ctx.brain.detect_symbol(msg_text, coin_map)
        if found_symbol:
            pot_pair = f"{found_symbol.lower()}usdt"
            detected_pairs.append(pot_pair)

    if detected_pairs is None or len(detected_pairs) == 0:
        return # Hiç coin yoksa geç

    # --- 3. ANALİZ DÖNGÜSÜ ---
    for pair in detected_pairs:
        try:
            # A) Geçmiş Veri Çekme (Haber anındaki 1 saatlik veri)
            klines = await ctx.real_exchange.client.futures_klines(
                symbol=pair.upper(),
                interval='1m',
                startTime=int(msg_ts * 1000),
                limit=61 # Analiz + 60dk takip
            )
            if not klines: continue

            # Haber anındaki fiyat (Entry)
            entry_price = float(klines[0][4]) # Close
            
            # Teknik Veriler
            tech = await get_historical_technicals(ctx, pair, msg_ts)
            if not tech: continue

            print(f"📊 Teknik Veriler Alındı ({pair}): RSI: {tech['rsi']:.2f} | BTC 1h: {tech['btc_trend']:.2f}%")

            # Güvenli Sözlük Erişimi ve Info
            clean_symbol = pair.lower().replace("usdt", "")
            c_data = coin_map.get(clean_symbol)
            if isinstance(c_data, dict):
                coin_full_name = c_data.get("name", "Unknown").title()
                m_cap = c_data.get("cap", 0)
            else:
                coin_full_name = "Unknown"
                m_cap = 0

            # Market Cap Formatlama
            if m_cap > 1_000_000_000:
                cap_str = f"${m_cap / 1_000_000_000:.2f} BILLION"
            elif m_cap > 1_000_000:
                cap_str = f"${m_cap / 1_000_000:.2f} MILLION"
            else:
                cap_str = "UNKNOWN/SMALL"

            # [CHANGED] B) Local Model ile Karar Al
            # Nexus AI v2 Analysis
            #remove links from msg_text
            rsi = tech['rsi']
            momentum = tech['changes']["1h"]
            rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            mom_label = "BULLISH_MOM" if momentum > 0.5 else "BEARISH_MOM" if momentum < -0.5 else "FLAT"

            msg_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', msg_text)
            msg_text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', msg_text)
            formatted_input = f"[N] {msg_text} [C] {pair.upper()} [MC] {cap_str} [RSI] {rsi_label} [MOM] {mom_label} [F]0.0"
            analysis = ctx.local_brain.analyze(formatted_input)
            action = analysis["action"]
            confidence = analysis["confidence"]

            print(f"🧠 NEXUS AI Karar: symbol: {pair}, action: {action}, confidence: {confidence:.2f}%")

            # C) Karar Uygulama (Confidence >= 65 Check)
            if confidence >= 75 and action in ["LONG", "SHORT"]:
                
                # --- SİMÜLASYON İŞLEM AÇILIŞI ---
                trade_amount = 100
                leverage = 10
                
                # TP/SL - Model vermiyor, biz manuel/dinamik atıyoruz
                if action == "LONG":
                    tp_pct = 2.0
                    sl_pct = 1.0
                else:
                    tp_pct = 2.0
                    sl_pct = 1.0

                report_entry = (
                    f"\n{'='*60}\n"
                    f"🔔 YENİ İŞLEM TESPİTİ | {msg_dt}\n"
                    f"{'-'*60}\n"
                    f"📰 HABER: {msg_text.strip()}\n"
                    f"🎯 HEDEF: {pair.upper()} ({coin_full_name})\n"
                    f"💰 MCAP: {cap_str}\n"
                    f"📈 TEKNİK: RSI={rsi:.2f} ({rsi_label}) | MOM={momentum:.2f} ({mom_label})\n"
                    f"🧠 AI KARARI (NEXUS v2):\n"
                    f"   - Aksiyon: {action}\n"
                    f"   - Güven: %{confidence:.2f}\n"
                    f"   - Raw Probs: {analysis['probs']}\n"
                    f"{'-'*60}\n"
                )
                
                # İşlemi aç
                open_log, _ = ctx.exchange.open_position_test(
                    symbol=pair, side=action, price=entry_price,
                    tp_pct=tp_pct, sl_pct=sl_pct,
                    amount_usdt=100, leverage=leverage, validity=30,
                    app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
                )
                
                print(f"🚀 İşlem Açıldı: {pair} | {action}")

                # --- 4. POZİSYON TAKİBİ (15sn Ticks) ---
                for k in klines:
                    minute_ts = k[0] / 1000
                    ticks = [float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                    
                    for i, tick_price in enumerate(ticks):
                        current_ts = minute_ts + (i * 15)
                        res_log, _, sym, pnl, peak, _ = ctx.exchange.check_positions_test(
                            pair, tick_price, now_ts=current_ts
                        )
                        
                        if res_log:
                            close_dt = datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M:%S")
                            duration_min = (current_ts - msg_ts) / 60
                            report_exit = (
                                f"🏁 İŞLEM SONUCU ({close_dt}):\n"
                                f"   - Durum: {res_log}\n"
                                f"   - Giriş: {entry_price} | Çıkış: {tick_price}\n"
                                f"   - Kaldıraç: {leverage}x\n"
                                f"   - Süre: {duration_min:.1f} dk\n"
                                f"   - Kar/Zarar: {pnl:.2f} USDT\n"
                                f"   - Görülen En İyi Fiyat (Peak): {peak}\n"
                                f"{'='*60}\n"
                            )
                            
                            f_log.write(report_entry + report_exit)
                            f_log.flush()
                            print(f"✅ İşlem Tamamlandı: {pair} | PnL: {pnl:.2f}")
                            return # Bir sonraki habere geç

            # D) Fırsat Kaçtı mı? (HOLD durumu veya Düşük Güven)
            elif action == "HOLD":
                # Gelecek 20 dakikaya bak
                # klines[0] şu anki mum. klines[1:21] sonraki 20 mum.
                future_candles = klines[1:21]
                if future_candles:
                    max_price = max([float(k[2]) for k in future_candles]) # High
                    min_price = min([float(k[3]) for k in future_candles]) # Low
                    
                    # Entry price'a göre değişim
                    pct_change_up = ((max_price - entry_price) / entry_price) * 100
                    pct_change_down = ((min_price - entry_price) / entry_price) * 100
                    
                    missed_action = None
                    change_val = 0.0
                    
                    if pct_change_up >= 1.5:
                        missed_action = "LONG"
                        change_val = pct_change_up
                    elif pct_change_down <= -1.5:
                        missed_action = "SHORT"
                        change_val = pct_change_down # negatif olacak
                        
                    if missed_action:
                        lev_10_profit = abs(change_val) * 10
                        
                        missed_log = (
                            f"\n{'='*60}\n"
                            f"⚠️ FIRSAT KAÇTI (HOLD) | {msg_dt}\n"
                            f"{'-'*60}\n"
                            f"📰 HABER: {msg_text.strip()}\n"
                            f"🎯 HEDEF: {pair.upper()} ({coin_full_name})\n"
                            f"🧠 AI KARARI: {action} (Güven: %{confidence:.2f})\n"
                            f"📉 GERÇEKLEŞEN (20dk): %{change_val:.2f} ({missed_action} Yönlü)\n"
                            f"💸 10x ile KAÇAN FIRSAT: %{lev_10_profit:.2f} PnL\n"
                            f"📈 TEKNİK: RSI={rsi:.2f} | MOM={momentum:.2f}\n"
                            f"{'='*60}\n"
                        )
                        f_log.write(missed_log)
                        f_log.flush()
                        print(f"⚠️ Fırsat Kaçtı Loglandı: {pair} | {change_val:.2f}%")

        except Exception as e:
            print(f"⚠️ Simülasyon Hatası ({pair}): {e}")

async def process_offline_entry(entry, ctx, f_log):
    """
    Processes a single entry from the offline dataset.
    """
    try:
        msg_text = entry["msg_text"]
        msg_dt = entry["msg_dt"]
        pair = entry["pair"]
        coin_full_name = entry["coin_full_name"]
        cap_str = entry["cap_str"]
        technicals = entry["technicals"]
        outcomes = entry["outcomes"]
        msg_ts = entry["msg_ts"]
        
        # Identify params from technicals
        rsi = technicals["rsi"]
        momentum = technicals["momentum_1h"]
        rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        mom_label = "BULLISH_MOM" if momentum > 0.5 else "BEARISH_MOM" if momentum < -0.5 else "FLAT"

        # AI Prediction
        # Clean text
        msg_text_clean = clean_news_text(msg_text)
        formatted_input = f"[N] {msg_text_clean} [C] {pair.replace("USDT", "")} [MC] {cap_str} [RSI] {rsi_label} [MOM] {mom_label} [F]0.0"
        

        changes = {
            "1h": technicals["momentum_1h"]
        }
        if LOCAL_BRAIN_MODE:
            analysis = ctx.local_brain.predict(news_text=msg_text_clean, symbol=pair.replace("USDT", ""))
            action = analysis["decision"]
            confidence = analysis["confidence"]
        else:
            # Construct changes dict mock
            changes = {
                "1m": 0.0,
                "10m": 0.0,
                "1h": technicals.get("momentum_1h", 0.0),
                "24h": 0.0
            }
            
            analysis = await ctx.brain.analyze_specific_no_research(
                news=msg_text_clean,
                symbol=pair.replace("USDT", ""),
            )
            action = analysis.get("action", "HOLD")
            confidence = analysis.get("conviction_score", 0)

        print(f"🧠 NEXUS AI Karar: symbol: {pair}, action: {action}, confidence: {confidence:.2f}%")

        # Decision Logic
        if confidence >= 75 and action in ["LONG", "SHORT"]:
            entry_price = technicals["close"]
            leverage = 10
            
            report_entry = (
                f"\n{'='*60}\n"
                f"🔔 YENİ İŞLEM TESPİTİ | {msg_dt}\n"
                f"{'-'*60}\n"
                f"📰 HABER: {msg_text_clean}\n"
                f"🎯 HEDEF: {pair} ({coin_full_name})\n"
                f"💰 MCAP: {cap_str}\n"
                f"📈 TEKNİK: RSI={rsi:.2f} ({rsi_label}) | MOM={momentum:.2f} ({mom_label})\n"
                f"🧠 AI KARARI (NEXUS v2):\n"
                f"   - Aksiyon: {action}\n"
                f"   - Güven: %{confidence:.2f}\n"
                f"{'-'*60}\n"
            )

            # Check Outcomes using 'future_candles' logic or simplified max_gain
            # To be consistent with online 'tick' check, we should iterate future candles
            future_candles = outcomes.get("future_candles", [])
            
            # Simulation Loop
            res_log = None
            pnl = 0.0
            peak = 0.0
            
            # Use PaperExchange to simulate trade logic step-by-step
            # First open
            if LOCAL_BRAIN_MODE:
                ctx.exchange.open_position_test(
                    symbol=pair, side=action, price=entry_price,
                    tp_pct=2.0, sl_pct=1.0,
                    amount_usdt=100, leverage=leverage, validity=30,
                    app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
                )
                print(f"🚀 İşlem Açıldı: {pair} | {action}")
            else:
                ctx.exchange.open_position_test(
                    symbol=pair, side=action, price=entry_price,
                    tp_pct=abs(analysis['tp_pct']), sl_pct=0.8,
                    amount_usdt=100, leverage=leverage, validity=analysis['validity_minutes'],
                    app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
                )
                print(f"🚀 İşlem Açıldı: {pair} | {action}")
            
            for k in future_candles:
                 # k is {'ts':..., 'o':..., 'h':..., 'l':..., 'c':...}
                 # Convert to [ts (ms), o, h, l, c] logic for consistency or just use High/Low/Close checks
                 # PaperExchange expects ticks or HL checks.
                 # Let's use check_positions_test
                 
                 # Create pseudo ticks: Open -> High -> Low -> Close (Simulating spread)
                 minute_ts = k['ts']
                 ticks = [k['o'], k['h'], k['l'], k['c']]
                 
                 for i, tick_price in enumerate(ticks):
                    current_ts = minute_ts + (i * 15)
                    r_log, _, _, val_pnl, val_peak, _ = ctx.exchange.check_positions_test(
                        pair, tick_price, now_ts=current_ts
                    )
                    
                    if r_log:
                        res_log = r_log
                        pnl = val_pnl
                        peak = val_peak
                        close_dt = datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M:%S")
                        duration_min = (current_ts - msg_ts) / 60
                        
                        report_exit = (
                            f"🏁 İŞLEM SONUCU ({close_dt}):\n"
                            f"   - Durum: {res_log}\n"
                            f"   - Giriş: {entry_price} | Çıkış: {tick_price}\n"
                            f"   - Kaldıraç: {leverage}x\n"
                            f"   - Süre: {duration_min:.1f} dk\n"
                            f"   - Kar/Zarar: {pnl:.2f} USDT\n"
                            f"   - Görülen En İyi Fiyat (Peak): {peak}\n"
                            f"{'='*60}\n"
                        )
                        f_log.write(report_entry + report_exit)
                        f_log.flush()
                        print(f"✅ İşlem Tamamlandı: {pair} | PnL: {pnl:.2f}")
                        return

        # Missed Opportunity Check
        elif action == "HOLD":
            entry_price = technicals["close"]
            max_gain_20m = outcomes.get("max_gain_20m", 0.0)
            max_loss_20m = outcomes.get("max_loss_20m", 0.0)
            
            missed_action = None
            change_val = 0.0
            
            if max_gain_20m >= 1.5:
                missed_action = "LONG"
                change_val = max_gain_20m
            elif max_loss_20m <= -1.5:
                missed_action = "SHORT"
                change_val = max_loss_20m 
            
            if missed_action:
                lev_10_profit = abs(change_val) * 10
                missed_log = (
                    f"\n{'='*60}\n"
                    f"⚠️ FIRSAT KAÇTI (HOLD) | {msg_dt}\n"
                    f"{'-'*60}\n"
                    f"📰 HABER: {msg_text.strip()}\n"
                    f"🎯 HEDEF: {pair} ({coin_full_name})\n"
                    f"🧠 AI KARARI: {action} (Güven: %{confidence:.2f})\n"
                    f"📉 GERÇEKLEŞEN (20dk): %{change_val:.2f} ({missed_action} Yönlü)\n"
                    f"💸 10x ile KAÇAN FIRSAT: %{lev_10_profit:.2f} PnL\n"
                    f"📈 TEKNİK: RSI={rsi:.2f} | MOM={momentum:.2f}\n"
                    f"{'='*60}\n"
                )
                f_log.write(missed_log)
                f_log.flush()
                print(f"⚠️ Fırsat Kaçtı Loglandı: {pair} | {change_val:.2f}%")

    except Exception as e:
        print(f"⚠️ Offline Error: {e}")

async def run_simulation():
    print("🚀 NEXUS BACKTEST SİMÜLASYONU BAŞLIYOR (Local SetFit)...")
    
    # Context Hazırlığı
    ctx = BotContext()
    ctx.app_state = SharedState()
    ctx.memory = MockMemory()
    ctx.exchange = PaperExchange(1000.0)
    
    # [NEW] Local Model
    ctx.local_brain = local_brain
    
    # Brain'i yine de init ediyoruz (detect_symbol için)
    ctx.brain = AgentBrain(
        use_groqcloud=False,
        api_key=GROQCLOUD_API_KEY,
        groqcloud_model=GROQCLOUD_MODEL,
        use_gemini=False,
        google_api_key=GOOGLE_API_KEY,
        gemini_model=GEMINI_MODEL
    )
    
    # [OFFLINE MODE CHECK]
    path = os.path.realpath(__file__)
    dir = os.path.dirname(path)
    dir = dir.replace("src", "data")
    dir = dir.replace("training", "")
    offline_file = os.path.join(dir, "offline_test_data.json")

    results_file = "data/backtest_results_nexus_phi.txt"
    if not os.path.exists(results_file):
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(offline_file):
        print(f"📂 Offline Veri Dosyası Bulundu: {offline_file}")
        print("⚡ OFFLINE SİMÜLASYON MODUNA GEÇİLİYOR (İnternetsiz)...")
        
        with open(offline_file, "r", encoding="utf-8") as f_in:
             offline_data = json.load(f_in)
        
        with open(results_file, "a", encoding="utf-8") as f_out:
            f_out.write(f"\n--- OFFLINE SIMULATION RUN: {datetime.now()} ---\n")
            
            print(f"📊 Toplam {len(offline_data)} adet veri işlenecek...")
            for i, entry in enumerate(offline_data):
                if i % 10 == 0: print(f"Processing {i}/{len(offline_data)}...")
                await process_offline_entry(entry, ctx, f_out)
                
        print(f"--- ✅ OFFLINE SİMÜLASYON BİTTİ. Sonuçlar: {results_file} ---")
        return

    # [ONLINE MODE FALLBACK] - REMOVED
    print("⚠️ Offline veri bulunamadı! Lütfen önce test_dataset.py çalıştırıp veriyi üretin.")
    return

if __name__ == "__main__":
    # Run the simulation for the SetFit model
    asyncio.run(run_simulation())