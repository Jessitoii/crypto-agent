"""
NEXUS Performance Analytics & Statistical Auditor

This module parses raw backtest execution logs to extract PnL data 
and calculate key performance indicators (KPIs) such as Win Rate, 
Profit Factor, and Average Trade Displacement.

It provides a clinical report of the model's profitability and 
statistical reliability.
"""

import re
import os 

def analyze_backtest(file_path):
    """
    Parses a backtest log and generates a comprehensive statistical report.
    
    Args:
        file_path (str): Path to the target .txt results file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract PnL displacement via regex patterns (e.g., PnL: 1.45 USDT)
        pnl_matches = re.findall(r'PnL:\s*(-?[\d\.]+)\s*USDT', content)
        pnls = [float(p) for p in pnl_matches]
        
        if not pnls:
            print("❌ Diagnostic Failure: No valid trade records found in log.")
            return

        # --- CORE KPI CALCULATIONS ---
        total_trades = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        total_profit = sum(wins)
        total_loss = sum(losses)
        
        win_rate = (len(wins) / total_trades) * 100
        starting_balance = 1000.0 # Baseline simulation capital
        final_balance = starting_balance + total_pnl
        
        # Profit Factor: Gross Profit / Gross Loss (Standardized efficiency metric)
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        print("\n📊 --- NEXUS AI BACKTEST ANALYTICS REPORT ---")
        print(f"🔹 Execution Count: {total_trades}")
        print(f"✅ Successful Signals: {len(wins)}")
        print(f"🛑 Invalidation Signals: {len(losses)}")
        print(f"📈 Strategic Win Rate: {win_rate:.2f}%")
        print(f"💰 Aggregate PnL: {total_pnl:+.2f} USDT")
        print(f"🏁 Initial Capital: {starting_balance:.2f} USDT")
        print(f"🚀 Terminus Capital: {final_balance:.2f} USDT")
        print("-" * 40)
        print(f"🏆 Max Drawdown Recovery (Peak): {max(pnls):+.2f} USDT")
        print(f"💀 Maximum Outlier Loss: {min(pnls):+.2f} USDT")
        print(f"⚖️ Mean Win Magnitude: {sum(wins)/len(wins) if wins else 0:+.2f} USDT")
        print(f"⚖️ Mean Loss Magnitude: {sum(losses)/len(losses) if losses else 0:+.2f} USDT")
        print(f"📊 Statistical Profit Factor: {profit_factor:.2f}")
        print("-" * 40)

    except FileNotFoundError:
        print(f"❌ IO Error: {file_path} not accessible.")
    except Exception as e:
        print(f"❌ Critical Reporting Failure: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_report = os.path.join(current_dir, 'backtest_results_nexus_qwen3.txt')
    analyze_backtest(target_report)
