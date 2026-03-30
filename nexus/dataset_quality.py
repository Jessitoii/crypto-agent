import json
from collections import Counter

def analyze_dataset(file_path):
    stats = Counter()
    total_samples = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sonuçları ekrana bas
        print("-" * 40)
        print(f"📊 DATASET ANALİZ RAPORU: {file_path}")
        print("-" * 40)
        print(f"Toplam Veri Sayısı: {total_samples}")
        print("-" * 40)
        
        for action, count in stats.items():
            percentage = (count / total_samples) * 100
            print(f"{action.ljust(10)}: {str(count).rjust(5)} adet (%{percentage:.2f})")
            
        print("-" * 40)
        
        # Mentör Kontrolü
        hold_ratio = (stats.get('HOLD', 0) / total_samples) * 100
        if hold_ratio < 70:
            print("⚠️ UYARI: HOLD oranı %70'in altında! Model hala her şeye atlıyor olabilir.")
        else:
            print("✅ TEBRİK: HOLD disiplini yüksek. Model gürültüyü elemeyi öğreniyor.")
            
    except FileNotFoundError:
        print("Hata: Dosya bulunamadı!")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

def check_logic_diversity(file_path):
    logics = []
    hold_count = 0
    long_count = 0
    short_count = 0
    # encoding='utf-8' ekleyerek Windows'un CP1254 dayatmasını engelliyoruz
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # hold long short sayısı
            hold_count = sum(1 for d in data if d['label'] == 0) 
            short_count = sum(1 for d in data if d['label'] == 1) 
            long_count = sum(1 for d in data if d['label'] == 2) 

            for d in data:
                if d['label'] == 0:
                    logics.append(d['reasoning'])
    except UnicodeDecodeError:
        # Eğer hala hata alırsan, utf-8-sig (BOM'lu dosyalar için) dene
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                d = json.loads(line)
                if d['label'] == 0:
                    logics.append(d['reasoning'])
    
    # Kelime analizi
    print("-" * 40)
    print(f"Hold Sayısı: {hold_count}")
    print(f"Long Sayısı: {long_count}")
    print(f"Short Sayısı: {short_count}")
    print("-" * 40)
    words = " ".join(logics).lower().split()
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    print("-" * 40)
    print("🔥 EN ÇOK TEKRARLANAN MANTIK KALIPLARI (TRİGRAMS):")
    for pattern, count in Counter(trigrams).most_common(10):
        print(f"{pattern.ljust(30)}: {count} kez")

    
check_logic_diversity('data/nexus_elite_v2.json')