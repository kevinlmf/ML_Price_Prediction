import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import yaml
import joblib
import os

class LSTMSentimentPredictor(nn.Module):
    def __init__(self, price_features: int, sentiment_features: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMSentimentPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Price LSTM layer
        self.price_lstm = nn.LSTM(
            input_size=price_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Sentiment LSTM layer  
        self.sentiment_lstm = nn.LSTM(
            input_size=sentiment_features,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size + hidden_size // 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        combined_size = hidden_size + hidden_size // 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # Binary classification: 0=下跌, 1=上涨
        )
        
        # Regression head for price change prediction
        self.regressor = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Predict price change percentage
        )
    
    def forward(self, price_seq, sentiment_seq):
        batch_size = price_seq.size(0)
        
        # Price LSTM
        price_out, (price_h, price_c) = self.price_lstm(price_seq)
        price_final = price_out[:, -1, :]  # Take last timestep
        
        # Sentiment LSTM  
        sent_out, (sent_h, sent_c) = self.sentiment_lstm(sentiment_seq)
        sent_final = sent_out[:, -1, :]
        
        # Combine features
        combined = torch.cat([price_final, sent_final], dim=1)
        
        # Apply attention (optional enhancement)
        combined_expanded = combined.unsqueeze(1)
        attn_out, _ = self.attention(combined_expanded, combined_expanded, combined_expanded)
        attn_final = attn_out.squeeze(1)
        
        # Predictions
        classification = self.classifier(attn_final)
        regression = self.regressor(attn_final)
        
        return classification, regression

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Scalers for normalization
        self.price_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        
    def prepare_data(self, price_sequences: np.ndarray, sentiment_sequences: np.ndarray, 
                    targets: np.ndarray) -> Tuple:
        """
        准备训练数据
        """
        # Normalize sequences
        batch_size, seq_len, price_features = price_sequences.shape
        _, _, sent_features = sentiment_sequences.shape
        
        # Reshape for scaling
        price_reshaped = price_sequences.reshape(-1, price_features)
        sent_reshaped = sentiment_sequences.reshape(-1, sent_features)
        
        # Fit and transform
        price_scaled = self.price_scaler.fit_transform(price_reshaped)
        sent_scaled = self.sentiment_scaler.fit_transform(sent_reshaped)
        
        # Reshape back
        price_scaled = price_scaled.reshape(batch_size, seq_len, price_features)
        sent_scaled = sent_scaled.reshape(batch_size, seq_len, sent_features)
        
        # Convert to tensors
        price_tensor = torch.FloatTensor(price_scaled)
        sent_tensor = torch.FloatTensor(sent_scaled)
        target_tensor = torch.LongTensor(targets)
        
        return price_tensor, sent_tensor, target_tensor
    
    def create_data_loader(self, price_data, sent_data, targets, batch_size: int = 32, shuffle: bool = True):
        """
        创建PyTorch DataLoader
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(price_data, sent_data, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_model(self, train_price, train_sent, train_targets, 
                   val_price, val_sent, val_targets) -> Dict[str, Any]:
        """
        训练LSTM模型
        """
        # Model parameters
        price_features = train_price.shape[2]
        sent_features = train_sent.shape[2]
        
        # Initialize model
        model = LSTMSentimentPredictor(
            price_features=price_features,
            sentiment_features=sent_features,
            hidden_size=self.config['prediction'].get('hidden_size', 128),
            num_layers=self.config['prediction'].get('num_layers', 2),
            dropout=self.config['prediction'].get('dropout', 0.2)
        ).to(self.device)
        
        # Loss functions
        classification_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['prediction'].get('learning_rate', 0.001)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Data loaders
        batch_size = self.config['prediction'].get('batch_size', 32)
        train_loader = self.create_data_loader(train_price, train_sent, train_targets, batch_size)
        val_loader = self.create_data_loader(val_price, val_sent, val_targets, batch_size, shuffle=False)
        
        # Training loop
        epochs = self.config['prediction'].get('epochs', 100)
        best_val_loss = float('inf')
        train_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_price, batch_sent, batch_targets in train_loader:
                batch_price = batch_price.to(self.device)
                batch_sent = batch_sent.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                class_pred, reg_pred = model(batch_price, batch_sent)
                
                # Calculate losses
                class_loss = classification_criterion(class_pred, batch_targets)
                
                # For regression, we need target price changes (mock for now)
                reg_targets = torch.randn(batch_targets.size(0), 1).to(self.device)
                reg_loss = regression_criterion(reg_pred, reg_targets)
                
                # Combined loss
                total_loss = class_loss + 0.1 * reg_loss  # Weight regression loss less
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_true = []
            
            with torch.no_grad():
                for batch_price, batch_sent, batch_targets in val_loader:
                    batch_price = batch_price.to(self.device)
                    batch_sent = batch_sent.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    class_pred, reg_pred = model(batch_price, batch_sent)
                    
                    class_loss = classification_criterion(class_pred, batch_targets)
                    reg_targets = torch.randn(batch_targets.size(0), 1).to(self.device)
                    reg_loss = regression_criterion(reg_pred, reg_targets)
                    
                    total_loss = class_loss + 0.1 * reg_loss
                    val_loss += total_loss.item()
                    
                    # Collect predictions
                    predictions = torch.argmax(class_pred, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true.extend(batch_targets.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_true, val_predictions)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save history
            train_history['train_loss'].append(avg_train_loss)
            train_history['val_loss'].append(avg_val_loss)
            train_history['val_accuracy'].append(val_accuracy)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(model, 'best_model.pth')
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}')
                print(f'  Val Loss: {avg_val_loss:.4f}')
                print(f'  Val Accuracy: {val_accuracy:.4f}')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        return {
            'model': model,
            'train_history': train_history,
            'best_val_loss': best_val_loss
        }
    
    def evaluate_model(self, model, test_price, test_sent, test_targets) -> Dict[str, float]:
        """
        评估模型性能
        """
        model.eval()
        test_loader = self.create_data_loader(test_price, test_sent, test_targets, 
                                            batch_size=64, shuffle=False)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_price, batch_sent, batch_targets in test_loader:
                batch_price = batch_price.to(self.device)
                batch_sent = batch_sent.to(self.device)
                
                class_pred, _ = model(batch_price, batch_sent)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(class_pred, dim=1)
                predictions = torch.argmax(class_pred, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def save_model(self, model, filename: str):
        """
        保存模型和预处理器
        """
        os.makedirs('models/saved/', exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'price_features': model.price_lstm.input_size,
                'sentiment_features': model.sentiment_lstm.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers
            }
        }, f'models/saved/{filename}')
        
        # Save scalers
        joblib.dump(self.price_scaler, f'models/saved/price_scaler.pkl')
        joblib.dump(self.sentiment_scaler, f'models/saved/sentiment_scaler.pkl')
        
        print(f"Model saved to models/saved/{filename}")
    
    def load_model(self, filename: str) -> LSTMSentimentPredictor:
        """
        加载模型
        """
        checkpoint = torch.load(f'models/saved/{filename}', map_location=self.device)
        
        model = LSTMSentimentPredictor(
            **checkpoint['model_config']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Load scalers
        self.price_scaler = joblib.load(f'models/saved/price_scaler.pkl')
        self.sentiment_scaler = joblib.load(f'models/saved/sentiment_scaler.pkl')
        
        return model

if __name__ == "__main__":
    # Test model training with dummy data
    trainer = ModelTrainer()
    
    # Create dummy data
    batch_size, seq_len, price_features = 100, 30, 9
    sent_features = 2
    
    train_price = np.random.randn(batch_size, seq_len, price_features)
    train_sent = np.random.randn(batch_size, seq_len, sent_features)
    train_targets = np.random.randint(0, 2, batch_size)
    
    val_price = np.random.randn(50, seq_len, price_features)
    val_sent = np.random.randn(50, seq_len, sent_features)
    val_targets = np.random.randint(0, 2, 50)
    
    # Prepare data
    train_price_tensor, train_sent_tensor, train_target_tensor = trainer.prepare_data(
        train_price, train_sent, train_targets)
    val_price_tensor, val_sent_tensor, val_target_tensor = trainer.prepare_data(
        val_price, val_sent, val_targets)
    
    print("Starting model training...")
    result = trainer.train_model(
        train_price_tensor, train_sent_tensor, train_target_tensor,
        val_price_tensor, val_sent_tensor, val_target_tensor
    )
    
    print("Training completed!")
    print(f"Best validation loss: {result['best_val_loss']:.4f}")