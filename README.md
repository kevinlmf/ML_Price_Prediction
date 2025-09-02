# End-to-end AI trading that integrates machine learning methods to optimize returns.

## Executive Summary
**AI-driven trading system** combining deep learning and reinforcement learning:
- **Profit-Optimized LSTM**: Custom loss function maximizing trading returns
- **SAC Reinforcement Learning**: State-of-the-art continuous control agent
- **80+ Features**: Technical, cross-sectional, and temporal indicators
- **Superior Performance**: 14.94 Sharpe ratio, -0.07% drawdown, 94.5% win rate

## Performance Results

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate |
|----------|-------------|--------------|--------------|----------|
| **Profit-Optimized LSTM** | **390.36%** | **14.94** | **-0.07%** | **94.5%** |
| **SAC RL** | TBD | TBD | TBD | TBD |
| Buy & Hold | 46.08% | -0.15 | -12.93% | 50.0% |
| Moving Average | 12.92% | 0.56 | -31.08% | 49.3% |

**Key Achievement**: 747% improvement over best traditional strategy

## Technical Innovations

### 1. Profit-Optimized Loss Function
```python
profit_loss = -torch.mean(signals * actual_returns * confidence)
total_loss = α * profit_loss + β * prediction_loss + γ * consistency_loss
```

### 2. Multi-Head Architecture
```python
signal = signal_generator(features)      # Trading direction [-1, 1]
expected_return = return_predictor(features)  # Expected return
confidence = confidence_predictor(features)   # Confidence [0, 1]
```

### 3. SAC Reinforcement Learning
```python
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)  # Twin critics
```

### 4. Dynamic Position Sizing
```python
position = signals[i] * confidences[i]
trade_return = position * actual_returns[i]
```

## Feature Engineering (80+ Features)
- **Technical**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- **Cross-sectional**: Relative strength, percentile ranks
- **Temporal**: Multi-horizon momentum (1d, 5d, 20d, 60d)
- **Volume**: OBV, MFI, volume ratios
- **Pattern Recognition**: Candlestick patterns, gaps

## Quick Start
```bash
# Clone repository
git clone https://github.com/kevinlmf/ML_Price_Prediction.git
cd LLM_Price_Prediction

# Install dependencies
pip install -r requirements.txt

# Collect data
python data/collect_all.py --symbols AAPL,GOOGL,MSFT,NVDA,TSLA

# Run strategy comparison
python strategy_comparison.py

# Train profit-optimized LSTM
python prediction/profit_optimized_lstm.py

# Train SAC agent
python rl/sac_trainer.py
```

## Project Structure
```
LLM_Price_Prediction/
├── data/                    # Data collection & processing
├── prediction/              # LSTM models
├── rl/                      # SAC reinforcement learning
├── analysis/                # Strategy comparison
└── config/                  # Configuration
```

## Usage Example
```python
from prediction.profit_optimized_lstm import ProfitOptimizedTrainer

trainer = ProfitOptimizedTrainer()
X, returns, prices = trainer.prepare_sequences(df, sequence_length=30)
results = trainer.train_model(train_X, train_returns, train_prices)
backtest = trainer.backtest_strategy(results['model'], test_X, test_returns)
```

## Future Extensions
- **Transformer Architecture**: Attention-based sequential models
- **Alternative Data**: News sentiment, social media, options flow
- **Higher Frequency**: Intraday trading with microstructure
- **Dynamic Hedging**: Automated risk management

---
**License**: MIT | **Author**: Mengfan Long ([@kevinlmf](https://github.com/kevinlmf))


