import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.fitted_models = {}
        
    def prepare_features(self, price_sequences: np.ndarray, sentiment_sequences: np.ndarray) -> np.ndarray:
        """
        将序列数据转换为传统机器学习特征
        """
        batch_size, seq_len, price_features = price_sequences.shape
        _, _, sent_features = sentiment_sequences.shape
        
        features = []
        
        for i in range(batch_size):
            price_seq = price_sequences[i]
            sent_seq = sentiment_sequences[i]
            
            # Price-based features
            price_feature_vector = []
            
            # Latest values
            price_feature_vector.extend(price_seq[-1])  # Latest price data
            
            # Technical indicators from price sequence
            closes = price_seq[:, 3]  # Assuming close price is at index 3
            
            # Moving averages
            ma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
            ma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
            ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            
            price_feature_vector.extend([ma_5, ma_10, ma_20])
            
            # Volatility
            volatility = np.std(closes[-10:]) if len(closes) >= 10 else 0
            price_feature_vector.append(volatility)
            
            # Price momentum
            momentum_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
            momentum_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
            price_feature_vector.extend([momentum_5, momentum_10])
            
            # RSI approximation
            price_changes = np.diff(closes[-14:]) if len(closes) >= 14 else np.array([0])
            gains = price_changes[price_changes > 0]
            losses = -price_changes[price_changes < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.01
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            price_feature_vector.append(rsi)
            
            # Sentiment-based features
            sent_feature_vector = []
            
            # Latest sentiment values
            sent_feature_vector.extend(sent_seq[-1])
            
            # Sentiment statistics
            sent_scores = sent_seq[:, 0]  # Assuming sentiment score is first column
            sent_feature_vector.extend([
                np.mean(sent_scores),
                np.std(sent_scores),
                np.max(sent_scores),
                np.min(sent_scores),
                sent_scores[-1] - sent_scores[0],  # Sentiment change
            ])
            
            # Combine all features
            combined_features = price_feature_vector + sent_feature_vector
            features.append(combined_features)
        
        return np.array(features)
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        训练逻辑回归模型
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Validation predictions
        val_pred = model.predict(X_val_scaled)
        val_proba = model.predict_proba(X_val_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred, average='weighted'),
            'recall': recall_score(y_val, val_pred, average='weighted'),
            'f1_score': f1_score(y_val, val_pred, average='weighted')
        }
        
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        
        return metrics, val_pred, val_proba
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        训练随机森林模型
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred, average='weighted'),
            'recall': recall_score(y_val, val_pred, average='weighted'),
            'f1_score': f1_score(y_val, val_pred, average='weighted'),
            'feature_importance': model.feature_importances_
        }
        
        self.models['random_forest'] = model
        
        return metrics, val_pred, val_proba
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        训练支持向量机模型
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        val_pred = model.predict(X_val_scaled)
        val_proba = model.predict_proba(X_val_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred, average='weighted'),
            'recall': recall_score(y_val, val_pred, average='weighted'),
            'f1_score': f1_score(y_val, val_pred, average='weighted')
        }
        
        self.models['svm'] = model
        self.scalers['svm'] = scaler
        
        return metrics, val_pred, val_proba
    
    def train_naive_bayes(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        训练朴素贝叶斯模型
        """
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred, average='weighted'),
            'recall': recall_score(y_val, val_pred, average='weighted'),
            'f1_score': f1_score(y_val, val_pred, average='weighted')
        }
        
        self.models['naive_bayes'] = model
        
        return metrics, val_pred, val_proba
    
    def train_all_models(self, price_sequences: np.ndarray, sentiment_sequences: np.ndarray, 
                        targets: np.ndarray, train_ratio: float = 0.8):
        """
        训练所有基线模型
        """
        # Prepare features
        X = self.prepare_features(price_sequences, sentiment_sequences)
        y = targets
        
        # Train-test split
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        
        results = {}
        
        # Train all models
        print("Training Logistic Regression...")
        results['logistic_regression'] = self.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        print("Training Random Forest...")
        results['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val)
        
        print("Training SVM...")
        results['svm'] = self.train_svm(X_train, y_train, X_val, y_val)
        
        print("Training Naive Bayes...")
        results['naive_bayes'] = self.train_naive_bayes(X_train, y_train, X_val, y_val)
        
        return results, X_val, y_val
    
    def predict(self, model_name: str, price_sequences: np.ndarray, sentiment_sequences: np.ndarray):
        """
        使用指定模型进行预测
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        X = self.prepare_features(price_sequences, sentiment_sequences)
        
        model = self.models[model_name]
        
        # Apply scaling if needed
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities

class ARIMAModel:
    """
    ARIMA时间序列模型 (传统金融预测方法)
    """
    def __init__(self):
        self.models = {}
        self.fitted = False
    
    def check_stationarity(self, timeseries):
        """
        检查时间序列的平稳性
        """
        result = adfuller(timeseries)
        return result[1] <= 0.05  # p-value <= 0.05 means stationary
    
    def make_stationary(self, timeseries):
        """
        使时间序列平稳
        """
        if self.check_stationarity(timeseries):
            return timeseries, 0
        
        # Try first difference
        diff_ts = np.diff(timeseries)
        if self.check_stationarity(diff_ts):
            return diff_ts, 1
        
        # Try second difference
        diff2_ts = np.diff(diff_ts)
        return diff2_ts, 2
    
    def find_optimal_params(self, timeseries, max_p=5, max_d=2, max_q=5):
        """
        寻找最优的ARIMA参数
        """
        best_aic = float('inf')
        best_params = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(timeseries, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params
    
    def train_arima_models(self, price_data: dict, symbols: list):
        """
        为每个股票训练ARIMA模型
        """
        for symbol in symbols:
            if symbol in price_data:
                prices = price_data[symbol]['Close'].dropna().values
                
                # Find optimal parameters
                optimal_params = self.find_optimal_params(prices)
                
                if optimal_params:
                    try:
                        model = ARIMA(prices, order=optimal_params)
                        fitted_model = model.fit()
                        self.models[symbol] = fitted_model
                        print(f"ARIMA{optimal_params} trained for {symbol}")
                    except Exception as e:
                        print(f"Failed to train ARIMA for {symbol}: {e}")
        
        self.fitted = True
    
    def predict_direction(self, symbol: str, steps: int = 1):
        """
        预测价格方向
        """
        if not self.fitted or symbol not in self.models:
            return None
        
        model = self.models[symbol]
        forecast = model.forecast(steps=steps)
        
        # Get current price (last observation)
        current_price = model.model.endog[-1]
        
        # Predict direction (1 if up, 0 if down)
        predicted_price = forecast[0] if steps == 1 else forecast[-1]
        direction = 1 if predicted_price > current_price else 0
        
        return direction, predicted_price
    
    def evaluate_arima(self, price_data: dict, symbols: list, test_ratio: float = 0.2):
        """
        评估ARIMA模型性能
        """
        results = {}
        
        for symbol in symbols:
            if symbol in price_data and symbol in self.models:
                prices = price_data[symbol]['Close'].dropna().values
                test_size = int(len(prices) * test_ratio)
                
                # Use ARIMA model to predict test period
                predictions = []
                actuals = []
                
                for i in range(test_size):
                    try:
                        # Predict next price
                        pred_direction, pred_price = self.predict_direction(symbol)
                        actual_price = prices[-(test_size-i)]
                        prev_price = prices[-(test_size-i+1)]
                        
                        actual_direction = 1 if actual_price > prev_price else 0
                        
                        predictions.append(pred_direction)
                        actuals.append(actual_direction)
                        
                    except:
                        continue
                
                if predictions:
                    accuracy = accuracy_score(actuals, predictions)
                    results[symbol] = {
                        'accuracy': accuracy,
                        'predictions': predictions,
                        'actuals': actuals
                    }
        
        return results

if __name__ == "__main__":
    # Test baseline models with dummy data
    baseline = BaselineModels()
    
    # Create dummy data
    batch_size, seq_len, price_features = 200, 30, 9
    sent_features = 2
    
    price_sequences = np.random.randn(batch_size, seq_len, price_features)
    sentiment_sequences = np.random.randn(batch_size, seq_len, sent_features)
    targets = np.random.randint(0, 2, batch_size)
    
    print("Training all baseline models...")
    results, X_val, y_val = baseline.train_all_models(price_sequences, sentiment_sequences, targets)
    
    print("\nBaseline Model Results:")
    for model_name, (metrics, _, _) in results.items():
        print(f"{model_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
        print()