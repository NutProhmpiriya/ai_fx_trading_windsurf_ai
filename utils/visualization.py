import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_trade_history(trades_file, price_file=None):
    """
    Create interactive visualization of trade history with candlestick chart
    """
    # Load trade history
    trades_df = pd.read_csv(trades_file)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Load price data if available
    if price_file:
        price_df = pd.read_csv(price_file)
        price_df['time'] = pd.to_datetime(price_df['time'])
        price_df.set_index('time', inplace=True)
    else:
        return None, trades_df
    
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

    # Figure 2: Cumulative P&L
    fig2 = go.Figure()

    # Calculate and plot cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    fig2.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=trades_df['cumulative_pnl'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(width=2, color='blue')
    ))

    # Update layout for P&L chart
    fig2.update_layout(
        title='Cumulative Profit/Loss',
        yaxis_title='Profit/Loss ($)',
        xaxis_title='Date',
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        yaxis=dict(
            tickformat='.2f',  # 2 decimal places for USD
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        xaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        )
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
    }
    
    return [fig1, fig2], summary

def print_trade_summary(summary):
    """Print trading performance summary"""
    print("\nTrading Performance Summary:")
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
