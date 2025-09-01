#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML Trading Strategies vs Traditional Financial Strategies Comparison
Comparison of actual trading performance between ML models (LSTM, SAC) and traditional technical analysis strategies
"""

import sys
import os
sys.path.append('.')

from prediction.profit_optimized_lstm import ProfitOptimizedTrainer
from rl.sac_trainer import run_sac_experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple


class TraditionalStrategies:
    """Traditional financial strategies implementation"""
    
    def __init__(self):
        self.strategies = {}
    
    def buy_and_hold_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Buy and hold strategy"""
        results = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            if len(symbol_data) < 2:
                continue
                
            # Simple buy and hold: buy on first day, sell on last day
            initial_price = symbol_data.iloc[0]['Close']
            final_price = symbol_data.iloc[-1]['Close']
            total_return = (final_price - initial_price) / initial_price
            
            # Calculate daily returns
            daily_returns = symbol_data['Close'].pct_change().dropna()
            
            results.append({
                'symbol': symbol,
                'total_return': total_return,
                'daily_returns': daily_returns.values,
                'num_trades': 1,
                'sharpe_ratio': daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(daily_returns.values)
            })
        
        # Aggregate results from all stocks
        all_returns = np.concatenate([r['daily_returns'] for r in results])
        total_return = np.mean([r['total_return'] for r in results])
        
        return {
            'strategy_name': 'Buy & Hold',
            'total_return': total_return,
            'daily_returns': all_returns,
            'num_trades': len(results),
            'win_rate': np.mean([r['total_return'] > 0 for r in results]),
            'avg_return_per_trade': total_return / len(results) if results else 0,
            'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results]),
            'max_drawdown': np.mean([r['max_drawdown'] for r in results])
        }
    
    def moving_average_strategy(self, df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> Dict[str, Any]:
        """Moving average crossover strategy"""
        all_trades = []
        all_returns = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            if len(symbol_data) < long_window:
                continue
            
            # Calculate moving averages
            symbol_data['MA_short'] = symbol_data['Close'].rolling(window=short_window).mean()
            symbol_data['MA_long'] = symbol_data['Close'].rolling(window=long_window).mean()
            
            # Generate trading signals
            symbol_data['signal'] = 0
            symbol_data.loc[symbol_data['MA_short'] > symbol_data['MA_long'], 'signal'] = 1
            symbol_data.loc[symbol_data['MA_short'] <= symbol_data['MA_long'], 'signal'] = -1
            
            # Calculate trading returns
            symbol_data['position'] = symbol_data['signal'].shift(1)  # Based on previous day's signal
            symbol_data['daily_return'] = symbol_data['Close'].pct_change()
            symbol_data['strategy_return'] = symbol_data['position'] * symbol_data['daily_return']
            
            # Remove NaN values
            valid_data = symbol_data.dropna()
            
            if len(valid_data) > 0:
                strategy_returns = valid_data['strategy_return'].values
                all_returns.extend(strategy_returns)
                
                # Calculate number of trades (signal changes)
                signal_changes = (valid_data['signal'] != valid_data['signal'].shift(1)).sum()
                all_trades.append(signal_changes)
        
        if not all_returns:
            return self._empty_result('Moving Average')
        
        all_returns = np.array(all_returns)
        
        return {
            'strategy_name': 'Moving Average',
            'total_return': np.sum(all_returns),
            'daily_returns': all_returns,
            'num_trades': int(np.sum(all_trades)),
            'win_rate': np.mean(all_returns > 0),
            'avg_return_per_trade': np.mean(all_returns[all_returns != 0]) if np.any(all_returns != 0) else 0,
            'sharpe_ratio': all_returns.mean() / all_returns.std() * np.sqrt(252) if all_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(all_returns)
        }
    
    def rsi_strategy(self, df: pd.DataFrame, rsi_period: int = 14, oversold: int = 30, overbought: int = 70) -> Dict[str, Any]:
        """RSI overbought/oversold strategy"""
        all_trades = []
        all_returns = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            if len(symbol_data) < rsi_period + 1:
                continue
            
            # Use pre-calculated RSI data
            if 'rsi_14' in symbol_data.columns:
                rsi_values = symbol_data['rsi_14']
            else:
                continue
            
            # Generate trading signals
            symbol_data['signal'] = 0
            symbol_data.loc[rsi_values < oversold, 'signal'] = 1   # Oversold, buy
            symbol_data.loc[rsi_values > overbought, 'signal'] = -1  # Overbought, sell
            
            # Calculate trading returns
            symbol_data['position'] = symbol_data['signal'].shift(1)
            symbol_data['daily_return'] = symbol_data['Close'].pct_change()
            symbol_data['strategy_return'] = symbol_data['position'] * symbol_data['daily_return']
            
            valid_data = symbol_data.dropna()
            
            if len(valid_data) > 0:
                strategy_returns = valid_data['strategy_return'].values
                all_returns.extend(strategy_returns)
                
                signal_changes = (valid_data['signal'] != valid_data['signal'].shift(1)).sum()
                all_trades.append(signal_changes)
        
        if not all_returns:
            return self._empty_result('RSI Strategy')
        
        all_returns = np.array(all_returns)
        
        return {
            'strategy_name': 'RSI Strategy',
            'total_return': np.sum(all_returns),
            'daily_returns': all_returns,
            'num_trades': int(np.sum(all_trades)),
            'win_rate': np.mean(all_returns > 0),
            'avg_return_per_trade': np.mean(all_returns[all_returns != 0]) if np.any(all_returns != 0) else 0,
            'sharpe_ratio': all_returns.mean() / all_returns.std() * np.sqrt(252) if all_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(all_returns)
        }
    
    def bollinger_bands_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Bollinger Bands strategy"""
        all_trades = []
        all_returns = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            # Use pre-calculated Bollinger Bands data
            if not all(col in symbol_data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                continue
            
            # Generate trading signals
            symbol_data['signal'] = 0
            symbol_data.loc[symbol_data['Close'] < symbol_data['bb_lower'], 'signal'] = 1   # Below lower band, buy
            symbol_data.loc[symbol_data['Close'] > symbol_data['bb_upper'], 'signal'] = -1  # Above upper band, sell
            
            # Calculate trading returns
            symbol_data['position'] = symbol_data['signal'].shift(1)
            symbol_data['daily_return'] = symbol_data['Close'].pct_change()
            symbol_data['strategy_return'] = symbol_data['position'] * symbol_data['daily_return']
            
            valid_data = symbol_data.dropna()
            
            if len(valid_data) > 0:
                strategy_returns = valid_data['strategy_return'].values
                all_returns.extend(strategy_returns)
                
                signal_changes = (valid_data['signal'] != valid_data['signal'].shift(1)).sum()
                all_trades.append(signal_changes)
        
        if not all_returns:
            return self._empty_result('Bollinger Bands')
        
        all_returns = np.array(all_returns)
        
        return {
            'strategy_name': 'Bollinger Bands',
            'total_return': np.sum(all_returns),
            'daily_returns': all_returns,
            'num_trades': int(np.sum(all_trades)),
            'win_rate': np.mean(all_returns > 0),
            'avg_return_per_trade': np.mean(all_returns[all_returns != 0]) if np.any(all_returns != 0) else 0,
            'sharpe_ratio': all_returns.mean() / all_returns.std() * np.sqrt(252) if all_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(all_returns)
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
            
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _empty_result(self, strategy_name: str) -> Dict[str, Any]:
        """Return empty result"""
        return {
            'strategy_name': strategy_name,
            'total_return': 0.0,
            'daily_returns': np.array([]),
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_return_per_trade': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }


def compare_strategies():
    """Compare performance of ML strategies (LSTM, SAC) vs traditional financial strategies"""
    print("=" * 70)
    print("ML Trading Strategies vs Traditional Financial Strategies Comparison")
    print("=" * 70)
    
    # Check data
    tech_data_path = "data/processed/technical_indicators_data.csv"
    if not os.path.exists(tech_data_path):
        print(f"Please run technical indicators calculation first: python data/technical_indicators.py")
        return
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(tech_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Number of symbols: {df['symbol'].nunique()}")
    print(f"Total records: {len(df)}")
    
    # 1. Run profit-optimized LSTM
    print("\n1. Running Profit-Optimized LSTM Model...")
    print("-" * 50)
    
    trainer = ProfitOptimizedTrainer()
    trainer.config['prediction']['epochs'] = 30  # Reduce training time for comparison
    
    X, returns, prices = trainer.prepare_sequences(df, 30)
    (train_X, train_returns, train_prices, 
     val_X, val_returns, val_prices, 
     test_X, test_returns, test_prices) = trainer.split_data(X, returns, prices)
    
    train_X_scaled, val_X_scaled, test_X_scaled = trainer.normalize_data(train_X, val_X, test_X)
    
    training_result = trainer.train_model(
        train_X_scaled, train_returns, train_prices,
        val_X_scaled, val_returns, val_prices
    )
    
    lstm_result = trainer.backtest_strategy(
        training_result['model'], 
        test_X_scaled, test_returns, test_prices
    )
    
    print(f"LSTM Model Results:")
    print(f"  Total Return: {lstm_result['total_return']*100:.2f}%")
    print(f"  Number of Trades: {lstm_result['num_trades']}")
    print(f"  Win Rate: {lstm_result['win_rate']*100:.1f}%")
    print(f"  Sharpe Ratio: {lstm_result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {lstm_result['max_drawdown']*100:.2f}%")
    
    # 2. Run SAC (Reinforcement Learning)
    print("\n2. Running SAC Reinforcement Learning Agent...")
    print("-" * 50)
    
    sac_result = run_sac_experiment(df)
    
    print(f"SAC Agent Results:")
    print(f"  Total Return: {sac_result['total_return']*100:.2f}%")
    print(f"  Number of Trades: {sac_result['num_trades']}")
    print(f"  Win Rate: {sac_result['win_rate']*100:.1f}%")
    print(f"  Sharpe Ratio: {sac_result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {sac_result['max_drawdown']*100:.2f}%")
    
    # 3. Run traditional strategies
    print("\n3. Running Traditional Financial Strategies...")
    print("-" * 50)
    
    traditional = TraditionalStrategies()
    
    # Use same test period data
    test_start_idx = len(train_X) + len(val_X)
    test_df = df.iloc[test_start_idx:test_start_idx + len(test_X)].copy()
    
    # Run various traditional strategies
    strategies_results = {}
    
    print("Running Buy & Hold strategy...")
    strategies_results['buy_hold'] = traditional.buy_and_hold_strategy(test_df)
    
    print("Running Moving Average strategy...")
    strategies_results['moving_avg'] = traditional.moving_average_strategy(test_df)
    
    print("Running RSI strategy...")
    strategies_results['rsi'] = traditional.rsi_strategy(test_df)
    
    print("Running Bollinger Bands strategy...")
    strategies_results['bollinger'] = traditional.bollinger_bands_strategy(test_df)
    
    # 4. Results comparison
    print("\n4. Strategy Comparison Results")
    print("=" * 70)
    
    # Create comparison table
    strategies_data = []
    
    # Add LSTM results
    strategies_data.append({
        'Strategy': 'LSTM (Profit-Optimized)',
        'Total Return': f"{lstm_result['total_return']*100:.2f}%",
        'Trades': lstm_result['num_trades'],
        'Win Rate': f"{lstm_result['win_rate']*100:.1f}%",
        'Sharpe Ratio': f"{lstm_result['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{lstm_result['max_drawdown']*100:.2f}%"
    })
    
    # Add SAC results
    strategies_data.append({
        'Strategy': 'SAC (Reinforcement Learning)',
        'Total Return': f"{sac_result['total_return']*100:.2f}%",
        'Trades': sac_result['num_trades'],
        'Win Rate': f"{sac_result['win_rate']*100:.1f}%",
        'Sharpe Ratio': f"{sac_result['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{sac_result['max_drawdown']*100:.2f}%"
    })
    
    # Add traditional strategy results
    for key, result in strategies_results.items():
        strategies_data.append({
            'Strategy': result['strategy_name'],
            'Total Return': f"{result['total_return']*100:.2f}%",
            'Trades': result['num_trades'],
            'Win Rate': f"{result['win_rate']*100:.1f}%",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{result['max_drawdown']*100:.2f}%"
        })
    
    # Print comparison table
    print(f"{'Strategy':<20} {'Total Return':<12} {'Trades':<8} {'Win Rate':<10} {'Sharpe':<8} {'Max DD':<10}")
    print("-" * 80)
    for data in strategies_data:
        print(f"{data['Strategy']:<20} {data['Total Return']:<12} {data['Trades']:<8} "
              f"{data['Win Rate']:<10} {data['Sharpe Ratio']:<8} {data['Max Drawdown']:<10}")
    
    # 5. Analysis conclusions
    print("\n5. Analysis Conclusions")
    print("-" * 50)
    
    lstm_return = lstm_result['total_return']
    sac_return = sac_result['total_return']
    best_traditional = max(strategies_results.values(), key=lambda x: x['total_return'])
    
    # Find best ML strategy
    best_ml_strategy = 'LSTM' if lstm_return >= sac_return else 'SAC'
    best_ml_return = max(lstm_return, sac_return)
    
    print(f"🎯 Return Comparison:")
    print(f"  LSTM Model Total Return: {lstm_return*100:.2f}%")
    print(f"  SAC Agent Total Return: {sac_return*100:.2f}%")
    print(f"  Best Traditional Strategy ({best_traditional['strategy_name']}): {best_traditional['total_return']*100:.2f}%")
    
    if best_ml_return > best_traditional['total_return']:
        improvement = (best_ml_return - best_traditional['total_return']) / abs(best_traditional['total_return']) * 100
        print(f"  Best ML Strategy ({best_ml_strategy}) Advantage: +{improvement:.1f}%")
    else:
        decline = (best_traditional['total_return'] - best_ml_return) / abs(best_ml_return) * 100
        print(f"  Traditional Strategy Advantage: +{decline:.1f}%")
    
    print(f"\n📊 Risk-Adjusted Returns (Sharpe Ratio):")
    print(f"  LSTM Model: {lstm_result['sharpe_ratio']:.2f}")
    print(f"  SAC Agent: {sac_result['sharpe_ratio']:.2f}")
    print(f"  Best Traditional Strategy: {best_traditional['sharpe_ratio']:.2f}")
    
    print(f"\n🛡️ Risk Control (Max Drawdown):")
    print(f"  LSTM Model: {lstm_result['max_drawdown']*100:.2f}%")
    print(f"  SAC Agent: {sac_result['max_drawdown']*100:.2f}%")
    print(f"  Traditional Strategies Average: {np.mean([r['max_drawdown'] for r in strategies_results.values()])*100:.2f}%")
    
    print(f"\n🔄 Trading Efficiency:")
    print(f"  LSTM Number of Trades: {lstm_result['num_trades']}")
    print(f"  SAC Number of Trades: {sac_result['num_trades']}")
    print(f"  Traditional Strategies Average: {np.mean([r['num_trades'] for r in strategies_results.values()]):.0f}")
    
    return lstm_result, sac_result, strategies_results


def visualize_comparison(lstm_result: Dict[str, Any], sac_result: Dict[str, Any], strategies_results: Dict[str, Any]):
    """Visualize comparison results"""
    print("\n5. Generating comparison charts...")
    
    # Set font for English
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML Strategies vs Traditional Financial Strategies Comparison', fontsize=16, fontweight='bold')
    
    # 1. Total return comparison
    strategy_names = ['LSTM (Profit-Optimized)', 'SAC (Reinforcement Learning)'] + [r['strategy_name'] for r in strategies_results.values()]
    total_returns = [lstm_result['total_return']*100, sac_result['total_return']*100] + [r['total_return']*100 for r in strategies_results.values()]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = axes[0,0].bar(range(len(strategy_names)), total_returns, color=colors[:len(strategy_names)])
    axes[0,0].set_title('Total Return Comparison (%)', fontweight='bold')
    axes[0,0].set_xticks(range(len(strategy_names)))
    axes[0,0].set_xticklabels(strategy_names, rotation=45, ha='right')
    axes[0,0].set_ylabel('Return (%)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, total_returns):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Sharpe ratio comparison
    sharpe_ratios = [lstm_result['sharpe_ratio'], sac_result['sharpe_ratio']] + [r['sharpe_ratio'] for r in strategies_results.values()]
    bars = axes[0,1].bar(range(len(strategy_names)), sharpe_ratios, color=colors[:len(strategy_names)])
    axes[0,1].set_title('Sharpe Ratio Comparison', fontweight='bold')
    axes[0,1].set_xticks(range(len(strategy_names)))
    axes[0,1].set_xticklabels(strategy_names, rotation=45, ha='right')
    axes[0,1].set_ylabel('Sharpe Ratio')
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, value in zip(bars, sharpe_ratios):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Win rate comparison
    win_rates = [lstm_result['win_rate']*100, sac_result['win_rate']*100] + [r['win_rate']*100 for r in strategies_results.values()]
    bars = axes[1,0].bar(range(len(strategy_names)), win_rates, color=colors[:len(strategy_names)])
    axes[1,0].set_title('Win Rate Comparison (%)', fontweight='bold')
    axes[1,0].set_xticks(range(len(strategy_names)))
    axes[1,0].set_xticklabels(strategy_names, rotation=45, ha='right')
    axes[1,0].set_ylabel('Win Rate (%)')
    axes[1,0].set_ylim([0, 100])
    axes[1,0].grid(True, alpha=0.3)
    
    for bar, value in zip(bars, win_rates):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Max drawdown comparison
    max_drawdowns = [abs(lstm_result['max_drawdown'])*100, abs(sac_result['max_drawdown'])*100] + [abs(r['max_drawdown'])*100 for r in strategies_results.values()]
    bars = axes[1,1].bar(range(len(strategy_names)), max_drawdowns, color=colors[:len(strategy_names)])
    axes[1,1].set_title('Max Drawdown Comparison (%)', fontweight='bold')
    axes[1,1].set_xticks(range(len(strategy_names)))
    axes[1,1].set_xticklabels(strategy_names, rotation=45, ha='right')
    axes[1,1].set_ylabel('Max Drawdown (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    for bar, value in zip(bars, max_drawdowns):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    output_path = 'strategy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved as {output_path}")
    
    plt.show()


if __name__ == "__main__":
    lstm_result, sac_result, strategies_results = compare_strategies()
    
    # Automatically generate visualization chart
    print("\nGenerating comparison chart...")
    visualize_comparison(lstm_result, sac_result, strategies_results)