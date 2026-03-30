import re
import os 
def analyze_backtest(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # PnL verilerini çek (Örn: PnL: 1.45 USDT veya PnL: -0.42 USDT)
        pnl_matches = re.findall(r'PnL:\s*(-?[\d\.]+)\s*USDT', content)
        pnls = [float(p) for p in pnl_matches]
        
        if not pnls:
            print("❌ Dosyada işlenmiş işlem bulunamadı.")
            return

        # İstatistiksel Hesaplamalar
        total_trades = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        total_profit = sum(wins)
        total_loss = sum(losses)
        
        win_rate = (len(wins) / total_trades) * 100
        starting_balance = 1000.0
        final_balance = starting_balance + total_pnl
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        print("📊 --- NEXUS AI BACKTEST PERFORMANS RAPORU ---")
        print(f"🔹 Toplam İşlem Sayısı: {total_trades}")
        print(f"✅ Başarılı (Win): {len(wins)}")
        print(f"🛑 Hatalı (Loss): {len(losses)}")
        print(f"📈 Win Rate: %{win_rate:.2f}")
        print(f"💰 Toplam PnL: {total_pnl:+.2f} USDT")
        print(f"🏁 Başlangıç Bakiyesi: {starting_balance:.2f} USDT")
        print(f"🚀 Final Bakiye: {final_balance:.2f} USDT")
        print("-" * 40)
        print(f"🏆 En Büyük Kazanç: {max(pnls):+.2f} USDT")
        print(f"💀 En Büyük Kayıp: {min(pnls):+.2f} USDT")
        print(f"⚖️ Ortalama Kazanç: {sum(wins)/len(wins) if wins else 0:+.2f} USDT")
        print(f"⚖️ Ortalama Kayıp: {sum(losses)/len(losses) if losses else 0:+.2f} USDT")
        print(f"📊 Profit Factor: {profit_factor:.2f}")
        print("-" * 40)

    except FileNotFoundError:
        print(f"❌ Hata: {file_path} dosyası bulunamadı.")
    except Exception as e:
        print(f"❌ Beklenmedik bir hata oluştu: {e}")

if __name__ == "__main__":
    path = os.path.realpath(__file__)
    dir = os.path.dirname(path)
    analyze_backtest(dir + '\\backtest_results_nexus_qwen3.txt')
