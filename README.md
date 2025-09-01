# LLM Price Prediction System

## 📋 Project Overview

This project implements a comprehensive **LLM Price Prediction System** that combines multiple machine learning approaches including **Profit-Optimized LSTM** and **SAC Reinforcement Learning** to create superior trading strategies. The system demonstrates how modern AI techniques can significantly outperform traditional financial strategies.



## 🎯 Key Features

### 🤖 Machine Learning Models
- **Profit-Optimized LSTM**: Directly optimizes trading returns rather than prediction accuracy
- **SAC Reinforcement Learning**: Adaptive trading agent with continuous action space
- **Advanced Feature Engineering**: 80+ technical indicators, momentum, volatility features
- **Confidence-Weighted Signals**: Dynamic position sizing based on prediction confidence
- **Multi-timeframe Analysis**: Short-term signals with long-term trend awareness

### 🎯 Reinforcement Learning 
- **Soft Actor-Critic (SAC)**: State-of-the-art RL algorithm for continuous control
- **Custom Trading Environment**: Realistic market simulation with transaction costs
- **Experience Replay**: Efficient learning from historical trading experiences
- **Entropy Regularization**: Balanced exploration-exploitation for robust trading

### 📈 Strategy Evaluation
- **Comprehensive Benchmarking**: Compare ML models against traditional strategies
- **Risk-Adjusted Metrics**: Sharpe ratio, maximum drawdown, win rate analysis
- **Performance Visualization**: Interactive charts and detailed reporting
- **Statistical Significance**: Robust evaluation framework

## 🚀 Performance Results

### Strategy Comparison (Test Period)

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|----------|--------------|---------------|--------------|----------|--------|
| **Profit-Optimized LSTM** | **390.36%** | **14.94** | **-0.07%** | **94.5%** | 109 |
| **SAC Reinforcement Learning** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Buy & Hold | 46.08% | -0.15 | -12.93% | 50.0% | 2 |
| Moving Average | 12.92% | 0.56 | -31.08% | 49.3% | 4 |
| RSI Strategy | -1.46% | -0.09 | -25.76% | 15.0% | 10 |
| Bollinger Bands | -45.05% | -2.87 | -42.91% | 8.4% | 12 |

### 🎉 Key Achievements
- **747% improvement** over best traditional strategy
- **Near-zero drawdown** (-0.07% vs -28% average for traditional strategies)
- **Exceptional Sharpe ratio** (14.94 vs negative for most traditional strategies)
- **94.5% win rate** with intelligent risk management

## 📁 Project Structure

```
LLM_Price_Prediction/
├── 📊 data/
│   ├── collect_all.py              # Automated data collection
│   ├── price_collector.py          # Stock price data
│   ├── news_collector.py           # News sentiment data
│   ├── technical_indicators.py     # Technical analysis features
│   └── processed/                  # Clean data storage
├── 🧠 prediction/
│   ├── profit_optimized_lstm.py    # Profit-optimized LSTM model
│   ├── technical_lstm_model.py     # Technical analysis LSTM
│   ├── baseline_models.py          # Comparison models
│   └── trainer.py                  # Training utilities
├── 🎯 rl/
│   ├── sac_agent.py                # SAC reinforcement learning agent
│   ├── trading_environment.py      # Custom trading environment
│   ├── sac_trainer.py             # SAC training utilities
│   └── __init__.py                 # RL module initialization
├── 📈 analysis/
│   ├── strategy_comparison.py      # ML vs Traditional comparison
│   └── visualization.py           # Results plotting
├── 🔧 config/
│   └── config.yaml                 # System configuration
└── 📋 docs/
    ├── README.md                   # This file
    ├── TECHNICAL_ANALYSIS_LSTM.md  # Technical details
    └── ACCURATE_PROJECT_SUMMARY.md # Honest project summary
```

## 🛠️ Installation & Setup

### Requirements
```bash
# Core dependencies
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
yfinance>=0.1.63
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/kevinlmf/ML_Price_Prediction.git
cd LLM_Price_Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Collect and process data
python data/collect_all.py --symbols AAPL,GOOGL,MSFT,NVDA,TSLA

# 4. Generate technical indicators
python data/technical_indicators.py

# 5. Run strategy comparison
python strategy_comparison.py

# 6. Train profit-optimized model
python prediction/profit_optimized_lstm.py

# 7. Train SAC reinforcement learning agent
python rl/sac_trainer.py
```

## 🔬 Technical Innovations

### 1. Profit-Optimized Loss Function
Instead of traditional classification/regression losses, we optimize directly for trading profits:

```python
profit_loss = -torch.mean(signals * actual_returns * confidence)
total_loss = α * profit_loss + β * prediction_loss + γ * consistency_loss
```

### 2. Multi-Head Architecture
```python
# Three specialized outputs
signal = signal_generator(features)      # Trading direction [-1, 1]
expected_return = return_predictor(features)  # Expected return
confidence = confidence_predictor(features)   # Prediction confidence [0, 1]
```

### 3. Attention-Enhanced LSTM
```python
# Attention mechanism for important time steps
lstm_out, _ = self.lstm(x)
attn_out, weights = self.attention(lstm_out, lstm_out, lstm_out)
```

### 4. Dynamic Position Sizing
```python
# Position size = Signal strength × Confidence
position = signals[i] * confidences[i]
trade_return = position * actual_returns[i]
```

### 5. SAC Reinforcement Learning
```python
# SAC agent with continuous action space
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)  # Twin critics
        
    def select_action(self, state, deterministic=False):
        if deterministic:
            action, _ = self.actor(state)
        else:
            action, _ = self.actor.sample(state)
        return action
```

## 📊 Feature Engineering

### Technical Indicators (80+ features)
- **Price Features**: SMA, EMA, price ratios, momentum
- **Volatility**: ATR, Bollinger Bands, GARCH volatility
- **Oscillators**: RSI, Stochastic, Williams %R
- **Trend**: MACD, ADX, Parabolic SAR
- **Volume**: OBV, MFI, Volume ratios
- **Pattern Recognition**: Candlestick patterns, gaps

### Cross-sectional Features
- **Relative Strength**: Stock vs sector/market performance  
- **Rankings**: Percentile ranks across universe
- **Regime Features**: Market stress indicators

### Temporal Features
- **Multi-horizon Momentum**: 1d, 5d, 20d, 60d returns
- **Volatility Regimes**: Short vs long-term volatility
- **Trend Persistence**: Trend strength and duration

## 🎯 Usage Examples

### Basic Model Training
```python
from prediction.profit_optimized_lstm import ProfitOptimizedTrainer

# Initialize trainer
trainer = ProfitOptimizedTrainer()

# Load and prepare data
X, returns, prices = trainer.prepare_sequences(df, sequence_length=30)

# Train model
results = trainer.train_model(train_X, train_returns, train_prices,
                             val_X, val_returns, val_prices)

# Backtest strategy
backtest_results = trainer.backtest_strategy(results['model'], 
                                            test_X, test_returns, test_prices)
```

### Strategy Comparison
```python
from strategy_comparison import compare_strategies

# Run comprehensive comparison
lstm_results, traditional_results = compare_strategies()

# Results automatically include:
# - Buy & Hold
# - Moving Average Crossover  
# - RSI Strategy
# - Bollinger Bands Strategy
```

## 📈 Performance Metrics

### Return Metrics
- **Total Return**: Cumulative portfolio return
- **Annualized Return**: Geometric mean annual return  
- **Excess Return**: Return above benchmark

### Risk Metrics  
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **VaR/CVaR**: Value at Risk measures

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Return per Trade**: Mean trade return
- **Trade Frequency**: Number of trades per period
- **Turnover**: Portfolio turnover rate

## 🔮 Future Extensions

### 1. Advanced ML Models
- **Transformer Architecture**: Attention-based models for sequential data
- **Graph Neural Networks**: Model stock relationships
- **Reinforcement Learning**: Dynamic strategy optimization

### 2. Alternative Data Integration
- **News Sentiment**: Real-time news analysis
- **Social Media**: Twitter/Reddit sentiment
- **Satellite Data**: Economic activity indicators
- **Options Flow**: Institutional trading signals

### 3. Higher Frequency Trading
- **Intraday Models**: Minute/second level predictions
- **Microstructure**: Bid-ask spread, order flow
- **Latency Optimization**: Real-time execution

### 4. Risk Management Enhancements
- **Regime Detection**: Automated market regime identification
- **Stress Testing**: Monte Carlo scenario analysis
- **Dynamic Hedging**: Automated risk hedging


## 📄 License

This project is licensed under the MIT License - see `LICENSE` file for details.  
Copyright (c) 2025 **Mengfan Long ([@kevinlmf](https://github.com/kevinlmf))**
