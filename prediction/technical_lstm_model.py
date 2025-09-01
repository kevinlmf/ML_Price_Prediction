#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical Analysis LSTM Model

纯技术分析的LSTM模型，只使用价格和技术指标进行股票价格方向预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from typing import Tuple, Dict, Any, List
import yaml
import joblib
import os
from torch.utils.data import TensorDataset, DataLoader


class TechnicalLSTM(nn.Module):
    """
    基于技术指标的LSTM模型
    """
    
    def __init__(self, input_features: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, num_classes: int = 2):
        super(TechnicalLSTM, self).__init__()
        
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
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # 回归器 (可选，预测价格变化幅度)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用最后一个时间步的输出
        final_output = attn_out[:, -1, :]
        
        # 分类和回归预测
        classification = self.classifier(final_output)
        regression = self.regressor(final_output)
        
        return classification, regression, attn_weights


class TechnicalLSTMTrainer:
    """
    技术分析LSTM训练器
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # 默认配置
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
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备LSTM序列数据
        
        Args:
            df: 包含技术指标的DataFrame
            sequence_length: 序列长度
            
        Returns:
            X: 特征序列 (samples, sequence_length, features)
            y: 目标标签 (samples,)
        """
        # 选择技术指标特征列 (排除非数值列)
        exclude_cols = ['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price_Direction']
        self.feature_columns = [col for col in df.columns 
                               if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        print(f"Using {len(self.feature_columns)} technical features")
        
        X_list = []
        y_list = []
        
        # 按股票分组处理
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            if len(symbol_data) < sequence_length + 1:
                print(f"Warning: Not enough data for {symbol}: {len(symbol_data)} < {sequence_length + 1}")
                continue
            
            # 提取特征和目标
            features = symbol_data[self.feature_columns].values
            
            # 创建目标标签 (下一日价格方向)
            close_prices = symbol_data['Close'].values
            targets = np.diff(close_prices) > 0  # True=上涨, False=下跌
            targets = targets.astype(int)
            
            # 创建序列
            for i in range(len(features) - sequence_length):
                X_seq = features[i:i+sequence_length]
                y_label = targets[i+sequence_length-1]  # 预测序列末尾的下一天
                
                # 检查是否有NaN
                if not np.isnan(X_seq).any() and not np.isnan(y_label):
                    X_list.append(X_seq)
                    y_list.append(y_label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        分割数据集 (时间序列要按时间顺序分割)
        """
        n_samples = len(X)
        train_size = int(n_samples * self.config['prediction']['train_test_split'])
        val_size = int(n_samples * 0.1)  # 10% for validation
        
        # 按时间顺序分割
        train_X = X[:train_size]
        train_y = y[:train_size]
        
        val_X = X[train_size:train_size+val_size]  
        val_y = y[train_size:train_size+val_size]
        
        test_X = X[train_size+val_size:]
        test_y = y[train_size+val_size:]
        
        print(f"Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
        
        return train_X, train_y, val_X, val_y, test_X, test_y
    
    def normalize_data(self, train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray) -> Tuple:
        """
        标准化数据
        """
        # Reshape for scaling
        n_train, seq_len, n_features = train_X.shape
        train_X_reshaped = train_X.reshape(-1, n_features)
        
        # Fit scaler on training data
        train_X_scaled = self.scaler.fit_transform(train_X_reshaped)
        train_X_scaled = train_X_scaled.reshape(n_train, seq_len, n_features)
        
        # Transform validation and test data
        val_X_scaled = self.scaler.transform(val_X.reshape(-1, n_features)).reshape(val_X.shape)
        test_X_scaled = self.scaler.transform(test_X.reshape(-1, n_features)).reshape(test_X.shape)
        
        return train_X_scaled, val_X_scaled, test_X_scaled
    
    def create_data_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        """
        创建DataLoader
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_model(self, train_X: np.ndarray, train_y: np.ndarray,
                   val_X: np.ndarray, val_y: np.ndarray) -> Dict[str, Any]:
        """
        训练模型
        """
        # 模型初始化
        input_features = train_X.shape[2]
        model = TechnicalLSTM(
            input_features=input_features,
            hidden_size=self.config['prediction']['hidden_size'],
            num_layers=self.config['prediction']['num_layers'],
            dropout=self.config['prediction']['dropout']
        ).to(self.device)
        
        print(f"Model initialized with {input_features} input features")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 损失函数和优化器
        classification_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['prediction']['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 数据加载器
        batch_size = self.config['prediction']['batch_size']
        train_loader = self.create_data_loader(train_X, train_y, batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_X, val_y, batch_size, shuffle=False)
        
        # 训练历史
        train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        early_stopping_patience = 20
        
        # 训练循环
        epochs = self.config['prediction']['epochs']
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                class_pred, reg_pred, _ = model(batch_X)
                
                # 计算损失
                class_loss = classification_criterion(class_pred, batch_y)
                
                # 回归目标 (价格变化率)
                reg_targets = torch.randn(batch_y.size(0), 1).to(self.device) * 0.02  # Mock targets
                reg_loss = regression_criterion(reg_pred, reg_targets)
                
                total_loss = class_loss + 0.1 * reg_loss
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    class_pred, reg_pred, _ = model(batch_X)
                    
                    class_loss = classification_criterion(class_pred, batch_y)
                    reg_targets = torch.randn(batch_y.size(0), 1).to(self.device) * 0.02
                    reg_loss = regression_criterion(reg_pred, reg_targets)
                    
                    total_loss = class_loss + 0.1 * reg_loss
                    val_loss += total_loss.item()
                    
                    # 收集预测结果
                    predictions = torch.argmax(class_pred, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # 计算指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_targets, val_predictions)
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 记录历史
            train_history['train_loss'].append(avg_train_loss)
            train_history['val_loss'].append(avg_val_loss)
            train_history['val_accuracy'].append(val_accuracy)
            train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'feature_columns': self.feature_columns,
                    'scaler': self.scaler,
                    'config': self.config
                }, 'best_technical_lstm_model.pth')
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
                print(f'  Val Accuracy: {val_accuracy:.4f}')
                print(f'  Best Val Accuracy: {best_val_accuracy:.4f}')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        return {
            'model': model,
            'train_history': train_history,
            'best_val_accuracy': best_val_accuracy
        }
    
    def evaluate_model(self, model, test_X: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
        """
        评估模型性能
        """
        model.eval()
        test_loader = self.create_data_loader(test_X, test_y, batch_size=64, shuffle=False)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                
                class_pred, _, _ = model(batch_X)
                
                probabilities = torch.softmax(class_pred, dim=1)
                predictions = torch.argmax(class_pred, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # 分类报告
        report = classification_report(all_targets, all_predictions, 
                                     target_names=['Down', 'Up'], 
                                     output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'classification_report': report
        }


def run_technical_lstm_experiment():
    """
    运行技术分析LSTM实验
    """
    print("=== Technical Analysis LSTM Experiment ===")
    
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
    trainer = TechnicalLSTMTrainer()
    
    # 准备序列数据
    print("Preparing sequences...")
    sequence_length = trainer.config['prediction']['sequence_length']
    X, y = trainer.prepare_sequences(df, sequence_length=sequence_length)
    
    if len(X) == 0:
        print("No valid sequences created. Check your data quality.")
        return
    
    # 分割数据
    train_X, train_y, val_X, val_y, test_X, test_y = trainer.split_data(X, y)
    
    # 标准化数据
    train_X_scaled, val_X_scaled, test_X_scaled = trainer.normalize_data(train_X, val_X, test_X)
    
    # 训练模型
    print("Training model...")
    training_result = trainer.train_model(train_X_scaled, train_y, val_X_scaled, val_y)
    
    # 评估模型
    print("Evaluating model...")
    model = training_result['model']
    evaluation_result = trainer.evaluate_model(model, test_X_scaled, test_y)
    
    # 显示结果
    print("\n=== Final Results ===")
    print(f"Test Accuracy: {evaluation_result['accuracy']:.4f}")
    print(f"Test Precision: {evaluation_result['precision']:.4f}")
    print(f"Test Recall: {evaluation_result['recall']:.4f}")
    print(f"Test F1-Score: {evaluation_result['f1_score']:.4f}")
    
    print(f"\nBest Validation Accuracy: {training_result['best_val_accuracy']:.4f}")
    
    print("\nDetailed Classification Report:")
    report = evaluation_result['classification_report']
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{class_name:>10}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")


if __name__ == "__main__":
    run_technical_lstm_experiment()