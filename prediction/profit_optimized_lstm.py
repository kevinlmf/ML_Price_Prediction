#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profit-Optimized LSTM Model

LSTM model based on profit optimization, focused on maximizing trading returns rather than prediction accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any, List
import yaml
import joblib
import os
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict


class ProfitOptimizedLSTM(nn.Module):
    """
    Profit-optimized LSTM model
    Output: trading signal strength (continuous values) and expected returns
    """
    
    def __init__(self, input_features: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(ProfitOptimizedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_features = input_features
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 交易信号生成器 - 输出交易信号强度 [-1, 1]
        self.signal_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # 输出 [-1, 1], 负值做空，正值做多
        )
        
        # 收益预测器 - 预测下一期收益率
        self.return_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, 1)  # 预测收益率
        )
        
        # 信心度预测器 - 预测信号的可靠性
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # 输出 [0, 1], 表示信心度
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用最后一个时间步的输出
        final_output = attn_out[:, -1, :]
        
        # 生成三个输出
        signal = self.signal_generator(final_output)  # 交易信号 [-1, 1]
        expected_return = self.return_predictor(final_output)  # 预期收益率
        confidence = self.confidence_predictor(final_output)  # 信心度 [0, 1]
        
        return signal, expected_return, confidence, attn_weights


class ProfitLoss(nn.Module):
    """
    基于收益优化的损失函数
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super(ProfitLoss, self).__init__()
        self.alpha = alpha  # 收益损失权重
        self.beta = beta    # 信号一致性权重
        self.gamma = gamma  # 信心度正则化权重
        
    def forward(self, signals, predicted_returns, confidence, actual_returns):
        """
        计算基于收益优化的损失
        
        Args:
            signals: 交易信号 [-1, 1]
            predicted_returns: 预测收益率
            confidence: 信心度 [0, 1]
            actual_returns: 实际收益率
        """
        # 1. 收益损失：最大化 signal * actual_return
        # 如果信号和实际收益同号，损失小；异号损失大
        profit_loss = -torch.mean(signals.squeeze() * actual_returns * confidence.squeeze())
        
        # 2. 收益预测损失：预测收益率的MSE
        return_prediction_loss = nn.MSELoss()(predicted_returns.squeeze(), actual_returns)
        
        # 3. 信号一致性损失：信号方向应与预测收益方向一致
        signal_consistency_loss = torch.mean(
            torch.abs(torch.sign(signals.squeeze()) - torch.sign(predicted_returns.squeeze()))
        )
        
        # 4. 信心度正则化：避免过于自信
        confidence_reg = torch.mean(torch.abs(confidence.squeeze() - 0.5))
        
        total_loss = (self.alpha * profit_loss + 
                     self.beta * return_prediction_loss + 
                     0.3 * signal_consistency_loss + 
                     self.gamma * confidence_reg)
        
        return total_loss, {
            'profit_loss': profit_loss.item(),
            'return_loss': return_prediction_loss.item(),
            'consistency_loss': signal_consistency_loss.item(),
            'confidence_reg': confidence_reg.item()
        }


class ProfitOptimizedTrainer:
    """
    收益优化的LSTM训练器
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'prediction': {
                    'sequence_length': 30,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'batch_size': 32,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'train_test_split': 0.8
                }
            }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备收益优化的序列数据
        
        Returns:
            X: 特征序列 (samples, sequence_length, features)
            returns: 实际收益率 (samples,)
            prices: 价格信息用于回测 (samples, 2) [close_price, next_close_price]
        """
        exclude_cols = ['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price_Direction']
        self.feature_columns = [col for col in df.columns 
                               if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        print(f"Using {len(self.feature_columns)} technical features")
        
        X_list = []
        returns_list = []
        prices_list = []
        
        # 按股票分组处理
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            if len(symbol_data) < sequence_length + 1:
                print(f"Warning: Not enough data for {symbol}: {len(symbol_data)} < {sequence_length + 1}")
                continue
            
            # 提取特征
            features = symbol_data[self.feature_columns].values
            close_prices = symbol_data['Close'].values
            
            # 计算收益率
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # 创建序列
            for i in range(len(features) - sequence_length):
                if i + sequence_length >= len(returns):
                    break
                    
                X_seq = features[i:i+sequence_length]
                return_target = returns[i+sequence_length-1]  # 序列末尾的下一期收益率
                price_info = [close_prices[i+sequence_length-1], close_prices[i+sequence_length]]
                
                # 检查是否有NaN
                if not np.isnan(X_seq).any() and not np.isnan(return_target):
                    X_list.append(X_seq)
                    returns_list.append(return_target)
                    prices_list.append(price_info)
        
        X = np.array(X_list)
        returns = np.array(returns_list)
        prices = np.array(prices_list)
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        print(f"Return statistics: mean={returns.mean():.4f}, std={returns.std():.4f}")
        print(f"Positive returns: {(returns > 0).sum()} ({(returns > 0).mean()*100:.1f}%)")
        
        return X, returns, prices
    
    def split_data(self, X: np.ndarray, returns: np.ndarray, prices: np.ndarray) -> Tuple:
        """分割数据集"""
        n_samples = len(X)
        train_size = int(n_samples * self.config['prediction']['train_test_split'])
        val_size = int(n_samples * 0.1)
        
        # 按时间顺序分割
        train_X = X[:train_size]
        train_returns = returns[:train_size]
        train_prices = prices[:train_size]
        
        val_X = X[train_size:train_size+val_size]
        val_returns = returns[train_size:train_size+val_size]
        val_prices = prices[train_size:train_size+val_size]
        
        test_X = X[train_size+val_size:]
        test_returns = returns[train_size+val_size:]
        test_prices = prices[train_size+val_size:]
        
        print(f"Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
        
        return train_X, train_returns, train_prices, val_X, val_returns, val_prices, test_X, test_returns, test_prices
    
    def normalize_data(self, train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray) -> Tuple:
        """标准化数据"""
        n_train, seq_len, n_features = train_X.shape
        train_X_reshaped = train_X.reshape(-1, n_features)
        
        train_X_scaled = self.scaler.fit_transform(train_X_reshaped)
        train_X_scaled = train_X_scaled.reshape(n_train, seq_len, n_features)
        
        val_X_scaled = self.scaler.transform(val_X.reshape(-1, n_features)).reshape(val_X.shape)
        test_X_scaled = self.scaler.transform(test_X.reshape(-1, n_features)).reshape(test_X.shape)
        
        return train_X_scaled, val_X_scaled, test_X_scaled
    
    def create_data_loader(self, X: np.ndarray, returns: np.ndarray, prices: np.ndarray, 
                          batch_size: int, shuffle: bool = True):
        """创建DataLoader"""
        X_tensor = torch.FloatTensor(X)
        returns_tensor = torch.FloatTensor(returns)
        prices_tensor = torch.FloatTensor(prices)
        
        dataset = TensorDataset(X_tensor, returns_tensor, prices_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_model(self, train_X: np.ndarray, train_returns: np.ndarray, train_prices: np.ndarray,
                   val_X: np.ndarray, val_returns: np.ndarray, val_prices: np.ndarray) -> Dict[str, Any]:
        """训练收益优化模型"""
        input_features = train_X.shape[2]
        model = ProfitOptimizedLSTM(
            input_features=input_features,
            hidden_size=self.config['prediction']['hidden_size'],
            num_layers=self.config['prediction']['num_layers'],
            dropout=self.config['prediction']['dropout']
        ).to(self.device)
        
        print(f"Model initialized with {input_features} input features")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 损失函数和优化器
        criterion = ProfitLoss(alpha=1.0, beta=0.5, gamma=0.1)
        optimizer = optim.Adam(model.parameters(), lr=self.config['prediction']['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 数据加载器
        batch_size = self.config['prediction']['batch_size']
        train_loader = self.create_data_loader(train_X, train_returns, train_prices, batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_X, val_returns, val_prices, batch_size, shuffle=False)
        
        # 训练历史
        train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_profit': [],
            'learning_rate': []
        }
        
        best_val_profit = -float('inf')
        patience_counter = 0
        early_stopping_patience = 20
        
        # 训练循环
        epochs = self.config['prediction']['epochs']
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_losses_detail = defaultdict(float)
            
            for batch_X, batch_returns, batch_prices in train_loader:
                batch_X = batch_X.to(self.device)
                batch_returns = batch_returns.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                signals, pred_returns, confidence, _ = model(batch_X)
                
                # 计算损失
                total_loss, loss_details = criterion(signals, pred_returns, confidence, batch_returns)
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                for key, value in loss_details.items():
                    train_losses_detail[key] += value
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_signals = []
            val_returns = []
            val_confidences = []
            
            with torch.no_grad():
                for batch_X, batch_returns, batch_prices in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_returns = batch_returns.to(self.device)
                    
                    signals, pred_returns, confidence, _ = model(batch_X)
                    
                    total_loss, _ = criterion(signals, pred_returns, confidence, batch_returns)
                    val_loss += total_loss.item()
                    
                    val_signals.extend(signals.cpu().numpy())
                    val_returns.extend(batch_returns.cpu().numpy())
                    val_confidences.extend(confidence.cpu().numpy())
            
            # 计算验证集收益
            val_signals = np.array(val_signals).flatten()
            val_returns = np.array(val_returns)
            val_confidences = np.array(val_confidences).flatten()
            
            # 计算加权收益 (信号 * 实际收益 * 信心度)
            weighted_returns = val_signals * val_returns * val_confidences
            val_profit = weighted_returns.mean()
            
            # 学习率调整
            scheduler.step(val_loss / len(val_loader))
            
            # 记录历史
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_history['train_loss'].append(avg_train_loss)
            train_history['val_loss'].append(avg_val_loss)
            train_history['val_profit'].append(val_profit)
            train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping based on profit
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'feature_columns': self.feature_columns,
                    'scaler': self.scaler,
                    'config': self.config
                }, 'best_profit_lstm_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # 打印进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}')
                print(f'  Val Loss: {avg_val_loss:.4f}')
                print(f'  Val Profit: {val_profit:.6f}')
                print(f'  Best Val Profit: {best_val_profit:.6f}')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        return {
            'model': model,
            'train_history': train_history,
            'best_val_profit': best_val_profit
        }
    
    def backtest_strategy(self, model, test_X: np.ndarray, test_returns: np.ndarray, 
                         test_prices: np.ndarray) -> Dict[str, Any]:
        """回测交易策略"""
        model.eval()
        test_loader = self.create_data_loader(test_X, test_returns, test_prices, 
                                            batch_size=64, shuffle=False)
        
        all_signals = []
        all_returns = []
        all_confidences = []
        all_prices = []
        
        with torch.no_grad():
            for batch_X, batch_returns, batch_prices in test_loader:
                batch_X = batch_X.to(self.device)
                
                signals, _, confidence, _ = model(batch_X)
                
                all_signals.extend(signals.cpu().numpy())
                all_returns.extend(batch_returns.numpy())
                all_confidences.extend(confidence.cpu().numpy())
                all_prices.extend(batch_prices.numpy())
        
        signals = np.array(all_signals).flatten()
        returns = np.array(all_returns)
        confidences = np.array(all_confidences).flatten()
        prices = np.array(all_prices)
        
        # 交易策略回测
        results = self._calculate_backtest_metrics(signals, returns, confidences, prices)
        
        return results
    
    def _calculate_backtest_metrics(self, signals: np.ndarray, actual_returns: np.ndarray, 
                                   confidences: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """计算回测指标"""
        # 设置交易阈值 - 降低阈值以产生更多交易
        confidence_threshold = 0.4  # 只有信心度超过40%才交易
        signal_threshold = 0.1      # 信号强度阈值
        
        # 输出调试信息
        print(f"Signal statistics: mean={signals.mean():.4f}, std={signals.std():.4f}, min={signals.min():.4f}, max={signals.max():.4f}")
        print(f"Confidence statistics: mean={confidences.mean():.4f}, std={confidences.std():.4f}, min={confidences.min():.4f}, max={confidences.max():.4f}")
        print(f"Signals > {signal_threshold}: {(np.abs(signals) > signal_threshold).sum()}/{len(signals)} ({(np.abs(signals) > signal_threshold).mean()*100:.1f}%)")
        print(f"Confidence > {confidence_threshold}: {(confidences > confidence_threshold).sum()}/{len(confidences)} ({(confidences > confidence_threshold).mean()*100:.1f}%)")
        
        # 生成交易决策
        trade_mask = (np.abs(signals) > signal_threshold) & (confidences > confidence_threshold)
        print(f"Final trade mask: {trade_mask.sum()}/{len(trade_mask)} trades ({trade_mask.mean()*100:.1f}%)")
        
        if trade_mask.sum() == 0:
            print("Warning: No trades generated with current thresholds")
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'buy_hold_return': 0.0,
                'excess_return': 0.0,
                'trade_returns': np.array([])
            }
        
        # 计算交易收益
        trade_returns = []
        positions = []
        
        for i in range(len(signals)):
            if trade_mask[i]:
                # 决定交易方向和仓位大小
                position = signals[i] * confidences[i]  # 位置 = 信号强度 * 信心度
                trade_return = position * actual_returns[i]
                
                trade_returns.append(trade_return)
                positions.append(position)
        
        trade_returns = np.array(trade_returns)
        positions = np.array(positions)
        
        # 计算累积收益
        cumulative_returns = np.cumprod(1 + trade_returns) - 1
        
        # 基准收益 (buy and hold)
        buy_hold_return = (prices[-1, 1] - prices[0, 0]) / prices[0, 0] if len(prices) > 0 else 0
        
        # 计算各种指标
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        num_trades = len(trade_returns)
        win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
        avg_return_per_trade = trade_returns.mean() if len(trade_returns) > 0 else 0
        
        # 夏普比率 (简化版)
        if len(trade_returns) > 1:
            sharpe_ratio = trade_returns.mean() / trade_returns.std() * np.sqrt(252)  # 年化
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        if len(cumulative_returns) > 0:
            running_max = np.maximum.accumulate(cumulative_returns + 1)
            drawdown = (cumulative_returns + 1 - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'trade_returns': trade_returns,
            'cumulative_returns': cumulative_returns,
            'positions': positions
        }


def run_profit_optimized_experiment():
    """运行收益优化的LSTM实验"""
    print("=== Profit-Optimized LSTM Experiment ===")
    
    # 检查技术指标数据是否存在
    tech_data_path = "data/processed/technical_indicators_data.csv"
    if not os.path.exists(tech_data_path):
        print(f"Technical indicators data not found at {tech_data_path}")
        print("Please run the technical indicators calculation first:")
        print("python data/technical_indicators.py")
        return
    
    # 加载数据
    print("Loading technical indicators data...")
    df = pd.read_csv(tech_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Loaded data: {len(df)} records, {len(df.columns)} columns")
    print(f"Symbols: {df['symbol'].unique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # 初始化训练器
    trainer = ProfitOptimizedTrainer()
    
    # 准备序列数据
    print("Preparing sequences...")
    sequence_length = trainer.config['prediction']['sequence_length']
    X, returns, prices = trainer.prepare_sequences(df, sequence_length=sequence_length)
    
    if len(X) == 0:
        print("No valid sequences created. Check your data quality.")
        return
    
    # 分割数据
    train_X, train_returns, train_prices, val_X, val_returns, val_prices, test_X, test_returns, test_prices = trainer.split_data(X, returns, prices)
    
    # 标准化数据
    train_X_scaled, val_X_scaled, test_X_scaled = trainer.normalize_data(train_X, val_X, test_X)
    
    # 训练模型
    print("Training profit-optimized model...")
    training_result = trainer.train_model(
        train_X_scaled, train_returns, train_prices,
        val_X_scaled, val_returns, val_prices
    )
    
    # 回测策略
    print("Running backtest...")
    model = training_result['model']
    backtest_result = trainer.backtest_strategy(model, test_X_scaled, test_returns, test_prices)
    
    # 显示结果
    print("\n=== Final Results ===")
    print(f"Best Validation Profit: {training_result['best_val_profit']:.6f}")
    print("\n=== Backtest Results ===")
    print(f"Total Return: {backtest_result['total_return']*100:.2f}%")
    print(f"Number of Trades: {backtest_result['num_trades']}")
    print(f"Win Rate: {backtest_result['win_rate']*100:.1f}%")
    print(f"Average Return per Trade: {backtest_result['avg_return_per_trade']*100:.3f}%")
    print(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_result['max_drawdown']*100:.2f}%")
    print(f"Buy & Hold Return: {backtest_result['buy_hold_return']*100:.2f}%")
    print(f"Excess Return: {backtest_result['excess_return']*100:.2f}%")
    
    return training_result, backtest_result


if __name__ == "__main__":
    run_profit_optimized_experiment()