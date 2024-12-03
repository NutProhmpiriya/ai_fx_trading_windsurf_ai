import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def calculate_monthly_stats(trades_df):
    """Calculate monthly trading statistics"""
    # Convert exit_time to datetime if it's not already
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['month'] = trades_df['exit_time'].dt.strftime('%Y-%m')

    # แยก Buy และ Sell orders
    buy_trades = trades_df[trades_df['trade_type'] == 'buy']
    sell_trades = trades_df[trades_df['trade_type'] == 'sell']

    # คำนวณสถิติรายเดือนสำหรับ Buy orders
    buy_monthly = buy_trades.groupby('month').agg({
        'pnl': ['count', 'sum', lambda x: (x > 0).sum(), lambda x: (x <= 0).sum()]
    })
    buy_monthly.columns = ['total_trades', 'net_profit', 'winning_trades', 'losing_trades']
    buy_monthly['win_rate'] = (buy_monthly['winning_trades'] / buy_monthly['total_trades'] * 100)
    buy_monthly['avg_profit_per_trade'] = buy_monthly['net_profit'] / buy_monthly['total_trades']

    # คำนวณสถิติรายเดือนสำหรับ Sell orders
    sell_monthly = sell_trades.groupby('month').agg({
        'pnl': ['count', 'sum', lambda x: (x > 0).sum(), lambda x: (x <= 0).sum()]
    })
    sell_monthly.columns = ['total_trades', 'net_profit', 'winning_trades', 'losing_trades']
    sell_monthly['win_rate'] = (sell_monthly['winning_trades'] / sell_monthly['total_trades'] * 100)
    sell_monthly['avg_profit_per_trade'] = sell_monthly['net_profit'] / sell_monthly['total_trades']

    # คำนวณสถิติรายเดือนรวม (สำหรับกราฟ)
    monthly_stats = trades_df.groupby('month').agg({
        'pnl': ['count', 'sum', lambda x: (x > 0).sum(), lambda x: (x <= 0).sum()]
    })
    monthly_stats.columns = ['total_trades', 'net_profit', 'winning_trades', 'losing_trades']
    monthly_stats['win_rate'] = (monthly_stats['winning_trades'] / monthly_stats['total_trades'] * 100)
    monthly_stats['avg_profit_per_trade'] = monthly_stats['net_profit'] / monthly_stats['total_trades']

    # Calculate cumulative account balance for each month
    trades_df['cumulative_balance'] = 10000 + trades_df['pnl'].cumsum()  # Starting balance $10,000
    monthly_balance = trades_df.groupby('month')['cumulative_balance'].last()
    monthly_stats['balance'] = monthly_balance  # Add account balance

    return monthly_stats, buy_monthly, sell_monthly

def calculate_drawdown(trades_df):
    """Calculate drawdown from cumulative P&L"""
    cumulative = trades_df['pnl'].cumsum()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / (rolling_max.abs() + 1e-9) * 100
    return drawdown

def analyze_trade_history(trades_file, price_file=None):
    """
    Create interactive visualization of trade history with candlestick chart
    """
    # Load trade history
    trades_df = pd.read_csv(trades_file)
    print("Columns in trades_df:", trades_df.columns.tolist())
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Calculate monthly statistics
    monthly_stats, buy_monthly, sell_monthly = calculate_monthly_stats(trades_df)
    
    # Calculate drawdown
    drawdown = calculate_drawdown(trades_df)
    
    # Load price data if available
    if price_file:
        price_df = pd.read_csv(price_file)
        price_df['time'] = pd.to_datetime(price_df['time'])
        price_df.set_index('time', inplace=True)
    else:
        return None, trades_df, monthly_stats

    # Create two separate figures
    # Figure 1: Price and Trade Points
    fig1 = go.Figure()

    # Add candlestick
    fig1.add_trace(go.Candlestick(x=price_df.index,
                                open=price_df['open'],
                                high=price_df['high'],
                                low=price_df['low'],
                                close=price_df['close'],
                                name='USDJPY'))

    # Add buy trades
    buy_trades = trades_df[trades_df['trade_type'] == 'buy']
    
    # Add lines connecting buy entry and exit points
    for _, trade in buy_trades.iterrows():
        fig1.add_trace(go.Scatter(
            x=[trade['entry_time'], trade['exit_time']],
            y=[trade['entry_price'], trade['exit_price']],
            mode='lines',
            line=dict(color='lightgreen', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add buy entry points
    fig1.add_trace(go.Scatter(
        x=buy_trades['entry_time'],
        y=buy_trades['entry_price'],
        mode='markers',
        name='Buy Entry',
        marker=dict(
            symbol='triangle-up',
            size=10,
            color='green',
            line=dict(width=1, color='darkgreen')
        )
    ))

    fig1.add_trace(go.Scatter(
        x=buy_trades['exit_time'],
        y=buy_trades['exit_price'],
        mode='markers',
        name='Buy Exit',
        marker=dict(
            symbol='x',
            size=10,
            color='darkgreen',
            line=dict(width=1)
        )
    ))

    # Add sell trades
    sell_trades = trades_df[trades_df['trade_type'] == 'sell']
    
    # Add lines connecting sell entry and exit points
    for _, trade in sell_trades.iterrows():
        fig1.add_trace(go.Scatter(
            x=[trade['entry_time'], trade['exit_time']],
            y=[trade['entry_price'], trade['exit_price']],
            mode='lines',
            line=dict(color='pink', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add sell entry points
    fig1.add_trace(go.Scatter(
        x=sell_trades['entry_time'],
        y=sell_trades['entry_price'],
        mode='markers',
        name='Sell Entry',
        marker=dict(
            symbol='triangle-down',
            size=10,
            color='red',
            line=dict(width=1, color='darkred')
        )
    ))

    fig1.add_trace(go.Scatter(
        x=sell_trades['exit_time'],
        y=sell_trades['exit_price'],
        mode='markers',
        name='Sell Exit',
        marker=dict(
            symbol='x',
            size=10,
            color='darkred',
            line=dict(width=1)
        )
    ))

    # Update layout for price chart
    fig1.update_layout(
        title='USDJPY Price Chart with Trade Points',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        height=900,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        yaxis=dict(
            tickformat='.3f',  # 3 decimal places for JPY
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        xaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        )
    )

    # Create P&L and Drawdown chart (fig2)
    fig2 = make_subplots(rows=4, cols=1, 
                        subplot_titles=('Cumulative Profit/Loss', 'Drawdown (%)', 'Monthly Statistics', 'Monthly Performance Table'),
                        vertical_spacing=0.1,
                        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]],
                        row_heights=[0.3, 0.2, 0.2, 0.3])

    # Add Cumulative P&L trace
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    fig2.add_trace(
        go.Scatter(
            x=trades_df['exit_time'],
            y=trades_df['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(width=2, color='blue')
        ),
        row=1, col=1
    )

    # Add Drawdown trace
    fig2.add_trace(
        go.Scatter(
            x=trades_df['exit_time'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(width=2, color='red')
        ),
        row=2, col=1
    )

    # Add Monthly Statistics bars
    # Buy orders
    fig2.add_trace(
        go.Bar(
            x=buy_monthly.index,
            y=buy_monthly['net_profit'],
            name='Buy Net Profit',
            marker_color=['red' if x < 0 else 'lightgreen' for x in buy_monthly['net_profit']]
        ),
        row=3, col=1
    )

    # Sell orders
    fig2.add_trace(
        go.Bar(
            x=sell_monthly.index,
            y=sell_monthly['net_profit'],
            name='Sell Net Profit',
            marker_color=['red' if x < 0 else 'darkgreen' for x in sell_monthly['net_profit']]
        ),
        row=3, col=1
    )

    # Add Win Rate lines
    fig2.add_trace(
        go.Scatter(
            x=buy_monthly.index,
            y=buy_monthly['win_rate'],
            name='Buy Win Rate %',
            mode='lines+markers',
            line=dict(color='blue', width=2),
            yaxis='y2'
        ),
        row=3, col=1
    )

    fig2.add_trace(
        go.Scatter(
            x=sell_monthly.index,
            y=sell_monthly['win_rate'],
            name='Sell Win Rate %',
            mode='lines+markers',
            line=dict(color='red', width=2, dash='dot'),
            yaxis='y2'
        ),
        row=3, col=1
    )

    # Create a combined table
    buy_formatted = buy_monthly.copy()
    sell_formatted = sell_monthly.copy()
    
    for df in [buy_formatted, sell_formatted]:
        df['net_profit'] = df['net_profit'].round(3)
        df['win_rate'] = df['win_rate'].round(2)
        df['avg_profit_per_trade'] = df['avg_profit_per_trade'].round(3)
    
    combined_table = go.Table(
        header=dict(
            values=[
                'Month',
                'Buy Trades', 'Buy Net Profit', 'Buy Wins', 'Buy Losses', 'Buy Win Rate %', 'Buy Avg Profit',
                'Sell Trades', 'Sell Net Profit', 'Sell Wins', 'Sell Losses', 'Sell Win Rate %', 'Sell Avg Profit'
            ],
            font=dict(size=11, color='white'),
            fill_color=['darkblue'] + ['royalblue']*6 + ['darkred']*6,
            align='center'
        ),
        cells=dict(
            values=[
                [idx.strftime('%Y-%m') if hasattr(idx, 'strftime') else str(idx) for idx in buy_formatted.index],
                buy_formatted['total_trades'],
                buy_formatted['net_profit'].round(3),
                buy_formatted['winning_trades'],
                buy_formatted['losing_trades'],
                buy_formatted['win_rate'].round(2),
                buy_formatted['avg_profit_per_trade'].round(6),
                sell_formatted['total_trades'],
                sell_formatted['net_profit'].round(3),
                sell_formatted['winning_trades'],
                sell_formatted['losing_trades'],
                sell_formatted['win_rate'].round(2),
                sell_formatted['avg_profit_per_trade'].round(6)
            ],
            font=dict(size=10),
            fill_color=[
                'white',
                'aliceblue', 'aliceblue', 'aliceblue', 'aliceblue', 'aliceblue', 'aliceblue',
                'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose'
            ],
            align='center'
        )
    )

    # Add the combined table
    fig2.add_trace(combined_table, row=4, col=1)

    # Update layout
    fig2.update_layout(
        height=1200,  # Adjust height for single table
        showlegend=True,
        title_text="Trading Performance Analysis",
    )
    
    # Update Y-axes
    fig2.update_yaxes(
        title_text='Profit/Loss ($)',
        tickformat='.2f',
        gridcolor='lightgrey',
        zerolinecolor='lightgrey',
        row=1, col=1
    )
    fig2.update_yaxes(
        title_text='Drawdown (%)',
        tickformat='.1f',
        gridcolor='lightgrey',
        zerolinecolor='lightgrey',
        row=2, col=1
    )
    fig2.update_yaxes(
        title_text='Monthly P&L ($)',
        tickformat='.2f',
        gridcolor='lightgrey',
        zerolinecolor='lightgrey',
        row=3, col=1
    )
    
    # Update X-axes
    fig2.update_xaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', row=1, col=1)
    fig2.update_xaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', row=2, col=1)
    fig2.update_xaxes(
        title_text='Month',
        gridcolor='lightgrey',
        zerolinecolor='lightgrey',
        tickangle=45,
        row=3,
        col=1
    )

    # Calculate trade summary
    summary = {
        'total_trades': len(trades_df),
        'profitable_trades': len(trades_df[trades_df['pnl'] > 0]),
        'losing_trades': len(trades_df[trades_df['pnl'] <= 0]),
        'total_profit': trades_df[trades_df['pnl'] > 0]['pnl'].sum(),
        'total_loss': abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()),
        'net_profit': trades_df['pnl'].sum(),
        'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100,
        'avg_profit_per_trade': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
        'avg_loss_per_trade': trades_df[trades_df['pnl'] <= 0]['pnl'].mean(),
        'max_drawdown': drawdown.min(),
        'monthly_stats': monthly_stats
    }
    
    return [fig1, fig2], summary

def print_trade_summary(summary):
    """Print trading performance summary"""
    print("\nOverall Trading Performance Summary:")
    print("-" * 40)
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Profitable Trades: {summary['profitable_trades']}")
    print(f"Losing Trades: {summary['losing_trades']}")
    print(f"Win Rate: {summary['win_rate']:.2f}%")
    print(f"Total Profit: ${summary['total_profit']:.2f}")
    print(f"Total Loss: ${summary['total_loss']:.2f}")
    print(f"Net Profit: ${summary['net_profit']:.2f}")
    print(f"Average Profit per Winning Trade: ${summary['avg_profit_per_trade']:.2f}")
    print(f"Average Loss per Losing Trade: ${summary['avg_loss_per_trade']:.2f}")
    print(f"Maximum Drawdown: {summary['max_drawdown']:.2f}%")
    
    print("\nMonthly Performance Summary:")
    print("-" * 40)
    print(summary['monthly_stats'].to_string())
