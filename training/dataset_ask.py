import json
import os
from groq import AsyncGroq
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio
import re
from google import genai
from google.genai import types

load_dotenv()

MODE = "GEMINI"

# --- AYARLAR ---
GROQ_API_KEY = os.getenv("GROQCLOUD_API_KEY")
INPUT_FILE = "data/hold_data.json"
OUTPUT_FILE = "data/hold_data_reasoning.json"
IRREVELANT_OUTPUT_FILE = "data/nexus_elite_v2_12_ultra_pure_groq_irrelevant.json"
MODEL = "llama-3.3-70b-versatile"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

client = AsyncGroq(api_key=GROQ_API_KEY)
gclient = genai.Client(api_key=GEMINI_API_KEY)

# --- YARDIMCI FONKSİYON: GÜVENLİ KAYIT ---
def save_progress(data, filename):
    """Veriyi diske yazar, veri kaybını önler."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush() # Buffer'ı zorla boşalt

async def check_relevance(news_text, label, symbol):
    """
    Haberin coin ile alakasını LLM'e sorar. Timeout mekanizması eklenmiştir.
    """
    action_map = {1: "SHORT (Price Drop)", 2: "LONG (Price Rise)", 0: "HOLD"}
    intended_action = action_map.get(label, "UNKNOWN")

    prompt = f"""
    You are a senior cryptocurrency analyst and financial logic auditor. Your task is to question the causality between the news presented to you and the action taken. Don't just do a simple word match; evaluate the fundamental weight of the news on the market.

    INPUTS:

    NEWS: {news_text}

    ACTION: {intended_action}

    COIN: {symbol}

    ANALYSIS PROTOCOL (Don't Be Robotic, Reason!):

    NARRATIVE WEIGHT: Does this news carry 'momentum' that will trigger market makers and bots? Is it a simple announcement or a structural change? "Which coin do you think will skyrocket?" polls or general comments on Twitter are noise. Distinguish the noise.

    CAUSAL DIRECTION: There must be a financial logic between the news and the action. If the news strengthens the coin, go LONG; if it weakens it, go SHORT. However, if the news is positive but the action is given as a SHORT order using 'technical indicators' (RSI, BTC Trend, etc.) as an excuse, this is not data measuring the impact of the news; it is technical analysis data. Consider anything that does not reflect the impact of the news as [IRRELEVANT].

    CHRONOLOGICAL TRAP: Is the news the CAUSE or the RESULT of the price movement? 'Bitcoin dropped to $90,000' is not news, it's a situation report. If the news describes a price movement that has already occurred and does not present a new catalyst, this data is not predictive.

    LOGICAL CONSISTENCY: Giving a 'LONG' order in anticipation of a bull market during events that are clearly negative (increased supply), such as 'Token Unlock', is a logical fallacy. The supply of goods puts price pressure on the market. Reject such data that contains 'optimism' but contradicts economic realities.

    DECISION MECHANISM: Choose only one of these two options:

    [RELEVANT]: The news is a direct and logical catalyst for the stated action. Without the news, this action would not be expected.

    [IRRELEVANT]: The link between the news and the action is weak, reversed, noisy, or the news is merely a report of consequences.

    ANSWER FORMAT: Just write [RELEVANT] or [IRRELEVANT]. Do not provide any other explanation.
    """
    
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            # --- KRİTİK EKLENTİ: TIMEOUT ---
            # asyncio.wait_for ile API çağrısını sarmalıyoruz.
            # 30 saniye içinde cevap gelmezse exception fırlatır ve retry mekanizması çalışır.
            
            if MODE == "GROQ":
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=10
                    ),
                    timeout=30.0 # 30 Saniye Timeout
                )
                response = completion.choices[0].message.content.strip()
                return "[RELEVANT]" in response

            elif MODE == "GEMINI":
                # Gemini synchronous client kullanıyorsun ama async loop içindeyiz.
                # Bloklamayı önlemek için thread içinde çalıştırmak daha sağlıklı ama
                # şimdilik basit timeout mantığı kuralım.
                # Google GenAI kütüphanesi kendi timeout parametresine sahip olabilir, 
                # ancak asyncio ile sarmalamak en garantisidir (blocking call olsa bile).
                
                # Not: gclient senkron çalışır, bu yüzden asyncio.to_thread kullanmalıyız.
                def run_gemini():
                    return gclient.models.generate_content(
                        model="gemma-3-27b-it", # Model ismini düzelttim, gemma-3 henüz stabil olmayabilir veya gemma-2-27b-it kastettin.
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0)
                    )

                res = await asyncio.wait_for(
                    asyncio.to_thread(run_gemini),
                    timeout=30.0
                )
                response = res.text.strip()
                return "[RELEVANT]" in response

        except asyncio.TimeoutError:
            print(f"⏰ [TIMEOUT] İstek zaman aşımına uğradı. Tekrar deneniyor ({retries+1}/{max_retries})")
            retries += 1
            await asyncio.sleep(2) # Biraz bekle

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                retries += 1
                ms_match = re.search(r"try again in (\d+)ms", error_msg)
                sec_match = re.search(r"try again in (\d+)s", error_msg)
                wait_time = 1.0
                if ms_match: wait_time = float(ms_match.group(1)) / 1000.0
                elif sec_match: wait_time = float(sec_match.group(1))
                wait_time += 0.5
                print(f"⏳ [RATE LIMIT] {wait_time:.2f}s bekleniyor... (Deneme {retries}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ [ERROR] {e}")
                retries += 1 # Diğer hatalarda da retry yapması için artırdım, yoksa sonsuz döngü olabilir.
                await asyncio.sleep(1)
                
    return False # Retries biterse False dön (Güvenli taraf)

async def process_dataset():
    # Mevcut çıktı dosyası varsa oradan devam et (Resume Capability)
    processed_count = 0
    perfected_data = []
    irrelevant_data = []
    # Eğer daha önce bir çıktı dosyası oluşturduysan onu yükle
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                perfected_data = json.load(f)
                # Basit bir mantıkla kaç tanesi işlendiğini tahmin edemeyiz çünkü filtreleme yapıyoruz.
                # Ancak kaynak veri setini sırayla işliyorsak, index takibi yapmak gerekir.
                # Şimdilik "Resume" mantığını karıştırmıyorum, sadece veri kaybını önlemeye odaklanıyorum.
        except:
            pass

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"🔍 Toplam {len(data)} entry inceleniyor...")
    
    try:
        relevant_count = 0
        irrelevant_count = 0
        # Tqdm'e 'initial' parametresi ekleyerek resume mantığı eklenebilir ama
        # şimdilik baştan başlatıp sürekli kaydetmeye odaklanalım.
        for i, entry in enumerate(tqdm(data)):
            
            # --- OPTİMİZASYON ---
            # Eğer bu entry zaten perfected_data içinde varsa (ID kontrolü vb.) atla
            # (Bu kısım senin veri yapına göre özelleştirilmeli, şimdilik pas geçiyorum)

            news_text = entry.get('text', '')
            symbol = news_text.split('[C]')[-1].strip() if '[C]' in news_text else "General"
            
            is_relevant = await check_relevance(news_text, entry['label'], symbol)
            
            if is_relevant:
                relevant_count += 1
                perfected_data.append(entry)
            else:
                irrelevant_count += 1
                irrelevant_data.append(entry)

            # --- CHECKPOINT SAVING ---
            # Her 10 işlemde bir veya son işlemde diske yaz
            
            save_progress(perfected_data, OUTPUT_FILE)
            save_progress(irrelevant_data, IRREVELANT_OUTPUT_FILE)
    except KeyboardInterrupt:
        print("\n🛑 [USER INTERRUPT] İşlem kullanıcı tarafından durduruldu!")
        print("💾 Mevcut ilerleme kaydediliyor...")
        save_progress(perfected_data, OUTPUT_FILE)
        save_progress(irrelevant_data, IRREVELANT_OUTPUT_FILE)
        print("✅ Kaydedildi. Çıkış yapılıyor.")
        return

    print(f"\n✅ İşlem tamamlandı!")
    print(f"📉 Orijinal: {len(data)} | ✨ Temizlenmiş: {relevant_count} | ❌ İrrelevant: {irrelevant_count}")
    f.close()

if __name__ == "__main__":
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        print("❌ Hata: API Key eksik!")
    else:
        try:
            asyncio.run(process_dataset())
            
        except KeyboardInterrupt:
            pass