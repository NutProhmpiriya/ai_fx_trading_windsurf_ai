import sys
import glob
import os
from difflib import SequenceMatcher
from utils.visualization import analyze_trade_history, print_trade_summary

def find_latest_file(pattern, search_dirs=[".", "data/raw"]):
    """
    ค้นหาไฟล์ล่าสุดที่ตรงกับ pattern จากหลายโฟลเดอร์
    """
    latest_file = None
    latest_time = 0
    
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
        
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern)
        
        if files:
            latest_in_dir = max(files, key=os.path.getmtime)
            mtime = os.path.getmtime(latest_in_dir)
            
            if mtime > latest_time:
                latest_time = mtime
                latest_file = latest_in_dir
    
    return latest_file

def find_most_similar_price_file(trades_filename):
    """
    ค้นหาไฟล์ price data ใน data/raw ที่มีชื่อคล้ายกับไฟล์ trades มากที่สุด
    """
    if not os.path.exists("data/raw"):
        return None
        
    # หาส่วนที่เป็นวันที่หรือ identifier จากชื่อไฟล์ trades
    base_name = os.path.basename(trades_filename)
    identifier = base_name.replace("trades_", "").replace(".csv", "")
    
    # ค้นหาไฟล์ทั้งหมดใน data/raw
    all_files = glob.glob(os.path.join("data/raw", "*.csv"))
    
    best_match = None
    highest_ratio = 0
    
    for file in all_files:
        if "trades" in file.lower():  # ข้ามไฟล์ trades
            continue
            
        file_name = os.path.basename(file)
        # เปรียบเทียบความคล้ายของชื่อไฟล์
        ratio = SequenceMatcher(None, identifier, file_name).ratio()
        
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = file
            
    return best_match if highest_ratio > 0.3 else None  # ต้องมีความคล้ายกันอย่างน้อย 30%

def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_trades.py <trades_csv_path> <price_data_path>")
        sys.exit(1)

    trades_csv_path = sys.argv[1]
    price_data_path = sys.argv[2]

    print(f"Analyzing trade history from: {trades_csv_path}")
    print(f"Using price data from: {price_data_path}")

    figs, summary = analyze_trade_history(trades_csv_path, price_data_path)
    print("\nOverall Trading Performance Summary:")
    print("----------------------------------------")
    print_trade_summary(summary)

    # Save the charts
    base_filename = os.path.splitext(os.path.basename(trades_csv_path))[0]
    figs[0].write_html(f"trade_analysis_price_{base_filename}.html")
    figs[1].write_html(f"trade_analysis_pnl_{base_filename}.html")

    print(f"\nPrice chart saved to: trade_analysis_price_{base_filename}.html")
    print(f"P&L chart saved to: trade_analysis_pnl_{base_filename}.html")

if __name__ == "__main__":
    main()
