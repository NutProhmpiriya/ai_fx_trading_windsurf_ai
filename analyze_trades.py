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

def format_summary(summary_dict):
    """Format summary dictionary into HTML table"""
    overall_stats = f"""
    <h3>Overall Performance</h3>
    <table class="stats-table">
        <tr>
            <td><strong>Total Trades:</strong></td>
            <td>{summary_dict['total_trades']}</td>
            <td><strong>Win Rate:</strong></td>
            <td>{summary_dict['win_rate']:.2f}%</td>
        </tr>
        <tr>
            <td><strong>Profitable Trades:</strong></td>
            <td>{summary_dict['profitable_trades']}</td>
            <td><strong>Losing Trades:</strong></td>
            <td>{summary_dict['losing_trades']}</td>
        </tr>
        <tr>
            <td><strong>Total Profit:</strong></td>
            <td>${summary_dict['total_profit']:.2f}</td>
            <td><strong>Total Loss:</strong></td>
            <td>${summary_dict['total_loss']:.2f}</td>
        </tr>
        <tr>
            <td><strong>Net Profit:</strong></td>
            <td>${summary_dict['net_profit']:.2f}</td>
            <td><strong>Max Drawdown:</strong></td>
            <td>{summary_dict['max_drawdown']:.2f}%</td>
        </tr>
        <tr>
            <td><strong>Avg Profit per Trade:</strong></td>
            <td>${summary_dict['avg_profit_per_trade']:.6f}</td>
            <td><strong>Avg Loss per Trade:</strong></td>
            <td>${summary_dict['avg_loss_per_trade']:.6f}</td>
        </tr>
    </table>

    <h3>Risk Metrics</h3>
    <table class="stats-table">
        <tr>
            <td><strong>Sharpe Ratio:</strong></td>
            <td>{summary_dict['risk_metrics']['sharpe_ratio']:.2f}</td>
            <td><strong>Risk/Reward Ratio:</strong></td>
            <td>{summary_dict['risk_metrics']['risk_reward_ratio']:.2f}</td>
        </tr>
        <tr>
            <td><strong>Max Consecutive Wins:</strong></td>
            <td>{summary_dict['risk_metrics']['max_consecutive_wins']}</td>
            <td><strong>Max Consecutive Losses:</strong></td>
            <td>{summary_dict['risk_metrics']['max_consecutive_losses']}</td>
        </tr>
        <tr>
            <td><strong>Average Win Size:</strong></td>
            <td>${summary_dict['risk_metrics']['avg_win_size']:.2f}</td>
            <td><strong>Average Loss Size:</strong></td>
            <td>${summary_dict['risk_metrics']['avg_loss_size']:.2f}</td>
        </tr>
        <tr>
            <td><strong>Average Holding Time:</strong></td>
            <td colspan="3">{summary_dict['risk_metrics']['avg_holding_time']}</td>
        </tr>
    </table>

    <h3>Trading Patterns</h3>
    <div class="pattern-tables">
        <div class="pattern-table">
            <h4>Performance by Hour</h4>
            <table class="stats-table">
                <tr>
                    <th>Hour</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Avg Profit</th>
                    <th>Total Profit</th>
                </tr>
                {generate_hourly_rows(summary_dict['pattern_analysis']['hourly_stats'])}
            </table>
        </div>
        
        <div class="pattern-table">
            <h4>Performance by Day</h4>
            <table class="stats-table">
                <tr>
                    <th>Day</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Avg Profit</th>
                    <th>Total Profit</th>
                </tr>
                {generate_daily_rows(summary_dict['pattern_analysis']['daily_stats'])}
            </table>
        </div>
    </div>

    <h3>Monthly Performance</h3>
    <table class="stats-table">
        <tr>
            <th>Month</th>
            <th>Total Trades</th>
            <th>Net Profit</th>
            <th>Win Rate</th>
            <th>Avg Profit/Trade</th>
            <th>Account Balance</th>
        </tr>
    """
    
    monthly_stats = summary_dict['monthly_stats']
    for month, stats in monthly_stats.iterrows():
        overall_stats += f"""
        <tr>
            <td>{month}</td>
            <td>{stats['total_trades']}</td>
            <td>${stats['net_profit']:.3f}</td>
            <td>{stats['win_rate']:.2f}%</td>
            <td>${stats['avg_profit_per_trade']:.6f}</td>
            <td>${stats['balance']:.2f}</td>
        </tr>
        """
    
    overall_stats += "</table>"
    return overall_stats

def generate_hourly_rows(hourly_stats):
    """Generate HTML rows for hourly statistics"""
    rows = ""
    for hour, stats in hourly_stats.iterrows():
        rows += f"""
        <tr>
            <td>{hour:02d}:00</td>
            <td>{stats['trades_count']}</td>
            <td>{stats['win_rate']:.2f}%</td>
            <td>${stats['avg_profit']:.6f}</td>
            <td>${stats['total_profit']:.2f}</td>
        </tr>
        """
    return rows

def generate_daily_rows(daily_stats):
    """Generate HTML rows for daily statistics"""
    rows = ""
    for day, stats in daily_stats.iterrows():
        rows += f"""
        <tr>
            <td>{day}</td>
            <td>{stats['trades_count']}</td>
            <td>{stats['win_rate']:.2f}%</td>
            <td>${stats['avg_profit']:.6f}</td>
            <td>${stats['total_profit']:.2f}</td>
        </tr>
        """
    return rows

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_trades.py <trades_csv_path> <price_data_path>")
        sys.exit(1)

    trades_csv_path = sys.argv[1]
    price_data_path = sys.argv[2]

    print(f"Analyzing trade history from: {trades_csv_path}")
    print(f"Using price data from: {price_data_path}")

    # Create output filename based on trades CSV name
    base_name = os.path.splitext(os.path.basename(trades_csv_path))[0]
    output_path = os.path.join('backtest_results', f'backtest_report_{base_name}.html')

    # Analyze trades and generate report
    figs, summary = analyze_trade_history(trades_csv_path, price_data_path)
    
    # Format summary as HTML
    formatted_summary = format_summary(summary)
    
    # Create a single HTML file with both charts and summary
    report_html = f"""
    <html>
    <head>
        <title>Trading Analysis Report</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px;
                line-height: 1.6;
            }}
            .summary {{ 
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .stats-table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            .stats-table td, .stats-table th {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .stats-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .stats-table th {{
                background-color: #4CAF50;
                color: white;
            }}
            .pattern-tables {{
                display: flex;
                justify-content: space-between;
                gap: 20px;
                margin: 20px 0;
            }}
            .pattern-table {{
                flex: 1;
            }}
            h3 {{
                color: #2C3E50;
                margin-top: 30px;
            }}
            h4 {{
                color: #34495E;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Trading Analysis Report</h1>
        <div class="summary">
            {formatted_summary}
        </div>
        <div id="price_chart">
            {figs[0].to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
        <div id="pnl_chart">
            {figs[1].to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
    </body>
    </html>
    """

    # Save the combined report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_html)

    print(f"\nComplete analysis report saved to: {output_path}")

if __name__ == "__main__":
    main()
