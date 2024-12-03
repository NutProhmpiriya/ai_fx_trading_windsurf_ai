import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar

def process_trades(trade_history):
    """Process trades once and cache results"""
    if len(trade_history) < 2:
        return pd.DataFrame()
    
    # Create trades DataFrame in one go
    trades_data = []
    current_position = None
    entry_price = 0
    entry_date = None
    
    for date, price, action in trade_history:
        if action == 1:  # Buy
            if current_position is None:
                current_position = 'buy'
                entry_price = price
                entry_date = date
            elif current_position == 'sell':
                trades_data.append({
                    'date': date,
                    'pnl': entry_price - price,
                    'type': 'sell',
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': price
                })
                current_position = None
        elif action == 2:  # Sell
            if current_position is None:
                current_position = 'sell'
                entry_price = price
                entry_date = date
            elif current_position == 'buy':
                trades_data.append({
                    'date': date,
                    'pnl': price - entry_price,
                    'type': 'buy',
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': price
                })
                current_position = None
    
    return pd.DataFrame(trades_data)

def create_price_chart(test_data):
    """Create main price chart with minimal data"""
    # Aggressive downsampling for price data
    sample_size = min(1000, len(test_data))
    step = len(test_data) // sample_size
    sampled_data = test_data.iloc[::step].copy()
    
    fig = make_subplots(rows=2, cols=2,
                       row_heights=[0.7, 0.3],
                       column_widths=[0.7, 0.3],
                       specs=[[{"colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "table"}]],
                       vertical_spacing=0.1,
                       horizontal_spacing=0.05)
    
    # Add candlesticks
    fig.add_trace(go.Candlestick(
        x=sampled_data.index,
        open=sampled_data['open'],
        high=sampled_data['high'],
        low=sampled_data['low'],
        close=sampled_data['close'],
        name='Price'
    ), row=1, col=1)
    
    return fig

def add_trade_lines(fig, trades_df):
    """Add trade lines efficiently"""
    if trades_df.empty:
        return
        
    # Add buy trades
    buy_trades = trades_df[trades_df['type'] == 'buy']
    if not buy_trades.empty:
        fig.add_trace(go.Scatter(
            x=buy_trades['entry_date'].tolist() + buy_trades['exit_date'].tolist(),
            y=buy_trades['entry_price'].tolist() + buy_trades['exit_price'].tolist(),
            mode='lines',
            line=dict(color='green', width=1),
            name='Buy Trades',
            showlegend=False
        ), row=1, col=1)
    
    # Add sell trades
    sell_trades = trades_df[trades_df['type'] == 'sell']
    if not sell_trades.empty:
        fig.add_trace(go.Scatter(
            x=sell_trades['entry_date'].tolist() + sell_trades['exit_date'].tolist(),
            y=sell_trades['entry_price'].tolist() + sell_trades['exit_price'].tolist(),
            mode='lines',
            line=dict(color='red', width=1),
            name='Sell Trades',
            showlegend=False
        ), row=1, col=1)

def add_cumulative_returns(fig, trades_df):
    """Add cumulative returns chart efficiently"""
    if trades_df.empty:
        return
        
    # Sort trades by date and calculate cumulative returns
    trades_df = trades_df.sort_values('date')
    trades_df['cumulative_returns'] = trades_df['pnl'].cumsum()
    
    fig.add_trace(go.Scatter(
        x=trades_df['date'],
        y=trades_df['cumulative_returns'],
        name='Cumulative Returns',
        line=dict(color='blue'),
        fill='tozeroy'
    ), row=2, col=1)

def calculate_monthly_stats(trades_df):
    """Calculate monthly statistics efficiently"""
    if trades_df.empty:
        return pd.DataFrame()
    
    trades_df['year_month'] = trades_df['date'].dt.strftime('%Y-%m')
    monthly_stats = []
    
    for year_month, group in trades_df.groupby('year_month'):
        year, month = year_month.split('-')
        month_name = calendar.month_abbr[int(month)]
        
        buy_trades = group[group['type'] == 'buy']
        sell_trades = group[group['type'] == 'sell']
        
        monthly_stats.append({
            'Month': f"{month_name} {year}",
            'Total': len(group),
            'Buy': len(buy_trades),
            'Sell': len(sell_trades),
            'Buy Win%': f"{(buy_trades['pnl'] > 0).mean() * 100:.1f}%" if len(buy_trades) > 0 else "0%",
            'Sell Win%': f"{(sell_trades['pnl'] > 0).mean() * 100:.1f}%" if len(sell_trades) > 0 else "0%",
            'Return': f"{group['pnl'].sum():.2f}",
            'Sharpe': f"{group['pnl'].mean() / group['pnl'].std():.2f}" if len(group) > 1 and group['pnl'].std() != 0 else "0"
        })
    
    return pd.DataFrame(monthly_stats)

def add_monthly_table(fig, monthly_stats):
    """Add monthly statistics table"""
    if not monthly_stats.empty:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(monthly_stats.columns),
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[monthly_stats[col] for col in monthly_stats.columns],
                    font=dict(size=10),
                    align="left"
                )
            ),
            row=2, col=2
        )

def calculate_key_metrics(trades_df):
    """Calculate key metrics efficiently"""
    if trades_df.empty:
        return {}
    
    total_trades = len(trades_df)
    buy_trades = trades_df[trades_df['type'] == 'buy']
    sell_trades = trades_df[trades_df['type'] == 'sell']
    
    buy_count = len(buy_trades)
    sell_count = len(sell_trades)
    buy_win_rate = (buy_trades['pnl'] > 0).mean() * 100 if buy_count > 0 else 0
    sell_win_rate = (sell_trades['pnl'] > 0).mean() * 100 if sell_count > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    
    return {
        'Total Trades': total_trades,
        'Buy Trades': f"{buy_count} ({buy_win_rate:.1f}% win)",
        'Sell Trades': f"{sell_count} ({sell_win_rate:.1f}% win)",
        'Total Return': f"{total_pnl:.2f}",
        'Avg Return': f"{(total_pnl/total_trades):.2f}" if total_trades > 0 else "0",
        'Sharpe Ratio': f"{trades_df['pnl'].mean()/trades_df['pnl'].std():.2f}" if len(trades_df) > 1 and trades_df['pnl'].std() != 0 else "N/A"
    }

def create_backtest_visualization(test_data, trade_history, model_name=""):
    """Create focused backtest visualization"""
    print("Processing trades...", flush=True)
    trades_df = process_trades(trade_history)
    
    print("Creating price chart...", flush=True)
    fig = create_price_chart(test_data)
    
    print("Adding trade lines...", flush=True)
    add_trade_lines(fig, trades_df)
    
    print("Adding cumulative returns...", flush=True)
    add_cumulative_returns(fig, trades_df)
    
    print("Calculating monthly statistics...", flush=True)
    monthly_stats = calculate_monthly_stats(trades_df)
    add_monthly_table(fig, monthly_stats)
    
    print("Calculating key metrics...", flush=True)
    metrics = calculate_key_metrics(trades_df)
    
    # Update layout
    title_text = f"Backtest Results" + (f" - {model_name}" if model_name else "")
    metrics_text = "<br>".join([f"{k}: {v}" for k, v in metrics.items()])
    
    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sub>{metrics_text}</sub>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        xaxis_title="",
        yaxis_title="Price",
        yaxis2_title="Cumulative Returns",
        height=900,
        width=1400
    )
    
    return fig, metrics
