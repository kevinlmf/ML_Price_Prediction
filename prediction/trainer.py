import os
import yaml
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .lstm_model import ModelTrainer as LSTMTrainer
from .baseline_models import BaselineModels, ARIMAModel

class ExperimentRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.exp_dir = f"experiments/{self.experiment_id}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/models", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/plots", exist_ok=True)
        
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Results will be saved to: {self.exp_dir}")
    
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载处理后的数据
        """
        try:
            # Load aligned data
            aligned_df = pd.read_csv("data/processed/aligned_data.csv")
            aligned_df['date'] = pd.to_datetime(aligned_df['date'])
            
            print(f"Loaded {len(aligned_df)} aligned records")
            print(f"Date range: {aligned_df['date'].min()} to {aligned_df['date'].max()}")
            print(f"Symbols: {aligned_df['symbol'].unique()}")
            
            # Create data aligner to process sequences
            from data.data_aligner import DataAligner
            aligner = DataAligner()
            
            # Create sequences for LSTM
            lookback = self.config['prediction']['lookback_window']
            sequences, sentiment_seqs, targets = aligner.create_sequences(
                aligned_df, 
                lookback_window=lookback,
                prediction_horizon=self.config['prediction']['prediction_horizon']
            )
            
            print(f"Created {len(sequences)} sequences")
            print(f"Price sequence shape: {sequences.shape}")
            print(f"Sentiment sequence shape: {sentiment_seqs.shape}")
            print(f"Target distribution: {np.bincount(targets)}")
            
            return sequences, sentiment_seqs, targets
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please run data collection and alignment first")
            return None, None, None
    
    def split_data(self, sequences: np.ndarray, sentiment_seqs: np.ndarray, 
                  targets: np.ndarray) -> Tuple:
        """
        分割数据为训练、验证和测试集
        """
        n_samples = len(sequences)
        
        # 时间序列数据应按时间顺序分割
        train_ratio = self.config['prediction']['train_test_split']
        val_ratio = 0.1  # 10% for validation
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_X = sequences[:train_end]
        train_sent = sentiment_seqs[:train_end] 
        train_y = targets[:train_end]
        
        val_X = sequences[train_end:val_end]
        val_sent = sentiment_seqs[train_end:val_end]
        val_y = targets[train_end:val_end]
        
        test_X = sequences[val_end:]
        test_sent = sentiment_seqs[val_end:]
        test_y = targets[val_end:]
        
        print(f"Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
        
        return (train_X, train_sent, train_y, 
                val_X, val_sent, val_y,
                test_X, test_sent, test_y)
    
    def train_lstm_model(self, train_data: Tuple, val_data: Tuple) -> Dict[str, Any]:
        """
        训练LSTM模型
        """
        print("Training LSTM-Sentiment model...")
        
        train_X, train_sent, train_y = train_data
        val_X, val_sent, val_y = val_data
        
        trainer = LSTMTrainer(config_path="config/config.yaml")
        
        # Prepare data
        train_price_tensor, train_sent_tensor, train_target_tensor = trainer.prepare_data(
            train_X, train_sent, train_y)
        val_price_tensor, val_sent_tensor, val_target_tensor = trainer.prepare_data(
            val_X, val_sent, val_y)
        
        # Train model
        result = trainer.train_model(
            train_price_tensor, train_sent_tensor, train_target_tensor,
            val_price_tensor, val_sent_tensor, val_target_tensor
        )
        
        # Save model
        trainer.save_model(result['model'], f"{self.experiment_id}_lstm_model.pth")
        
        return {
            'model': result['model'],
            'trainer': trainer,
            'train_history': result['train_history'],
            'best_val_loss': result['best_val_loss']
        }
    
    def train_baseline_models(self, sequences: np.ndarray, sentiment_seqs: np.ndarray, 
                             targets: np.ndarray) -> Dict[str, Any]:
        """
        训练基线模型
        """
        print("Training baseline models...")
        
        baseline = BaselineModels()
        results, X_val, y_val = baseline.train_all_models(
            sequences, sentiment_seqs, targets,
            train_ratio=self.config['prediction']['train_test_split']
        )
        
        # Save baseline models
        with open(f"{self.exp_dir}/baseline_models.pkl", 'wb') as f:
            pickle.dump(baseline, f)
        
        return {
            'models': baseline,
            'results': results,
            'validation_data': (X_val, y_val)
        }
    
    def evaluate_all_models(self, test_data: Tuple, lstm_result: Dict, 
                           baseline_result: Dict) -> Dict[str, Any]:
        """
        评估所有模型在测试集上的性能
        """
        print("Evaluating all models on test set...")
        
        test_X, test_sent, test_y = test_data
        evaluation_results = {}
        
        # Evaluate LSTM model
        lstm_trainer = lstm_result['trainer']
        lstm_model = lstm_result['model']
        
        # Prepare test data for LSTM
        test_price_tensor, test_sent_tensor, test_target_tensor = lstm_trainer.prepare_data(
            test_X, test_sent, test_y)
        
        lstm_metrics = lstm_trainer.evaluate_model(
            lstm_model, test_price_tensor, test_sent_tensor, test_target_tensor)
        
        evaluation_results['LSTM-Sentiment'] = lstm_metrics
        
        # Evaluate baseline models
        baseline_models = baseline_result['models']
        
        for model_name in ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']:
            try:
                test_pred, test_proba = baseline_models.predict(model_name, test_X, test_sent)
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics = {
                    'accuracy': accuracy_score(test_y, test_pred),
                    'precision': precision_score(test_y, test_pred, average='weighted'),
                    'recall': recall_score(test_y, test_pred, average='weighted'),
                    'f1_score': f1_score(test_y, test_pred, average='weighted'),
                    'predictions': test_pred,
                    'probabilities': test_proba
                }
                
                evaluation_results[model_name.replace('_', ' ').title()] = metrics
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        return evaluation_results
    
    def create_comparison_plots(self, evaluation_results: Dict[str, Any]):
        """
        创建模型比较图表
        """
        print("Creating comparison plots...")
        
        # Prepare data for plotting
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model_name, metrics in evaluation_results.items():
            models.append(model_name)
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Precision comparison
        bars2 = ax2.bar(models, precisions, color='lightcoral', alpha=0.7)
        ax2.set_title('Model Precision Comparison')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Recall comparison
        bars3 = ax3.bar(models, recalls, color='lightgreen', alpha=0.7)
        ax3.set_title('Model Recall Comparison')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # F1-score comparison
        bars4 = ax4.bar(models, f1_scores, color='gold', alpha=0.7)
        ax4.set_title('Model F1-Score Comparison')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/plots/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create overall performance radar chart
        self.create_radar_chart(evaluation_results)
    
    def create_radar_chart(self, evaluation_results: Dict[str, Any]):
        """
        创建雷达图显示模型综合性能
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data
        models = list(evaluation_results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, model_name in enumerate(models):
            values = [
                evaluation_results[model_name]['accuracy'],
                evaluation_results[model_name]['precision'],
                evaluation_results[model_name]['recall'],
                evaluation_results[model_name]['f1_score']
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.savefig(f"{self.exp_dir}/plots/performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, all_results: Dict[str, Any]):
        """
        保存实验结果
        """
        print("Saving experiment results...")
        
        # Create summary report
        summary = {
            'experiment_id': self.experiment_id,
            'config': self.config,
            'model_performance': {}
        }
        
        # Extract key metrics
        for model_name, metrics in all_results.items():
            summary['model_performance'][model_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'], 
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
        
        # Save summary
        with open(f"{self.exp_dir}/experiment_summary.yaml", 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Save detailed results
        with open(f"{self.exp_dir}/detailed_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        # Create results table
        results_df = pd.DataFrame(summary['model_performance']).T
        results_df.to_csv(f"{self.exp_dir}/model_comparison.csv")
        
        print(f"Results saved to {self.exp_dir}")
        print("\nModel Performance Summary:")
        print(results_df.round(4))
        
        # Find best model
        best_model = results_df['f1_score'].idxmax()
        best_f1 = results_df['f1_score'].max()
        
        print(f"\nBest Model: {best_model} (F1-Score: {best_f1:.4f})")
    
    def run_full_experiment(self):
        """
        运行完整的实验流程
        """
        print("Starting full experiment...")
        print("=" * 50)
        
        # 1. Load data
        sequences, sentiment_seqs, targets = self.load_processed_data()
        if sequences is None:
            print("Failed to load data. Experiment terminated.")
            return
        
        # 2. Split data
        (train_X, train_sent, train_y,
         val_X, val_sent, val_y,
         test_X, test_sent, test_y) = self.split_data(sequences, sentiment_seqs, targets)
        
        # 3. Train LSTM model
        lstm_result = self.train_lstm_model(
            (train_X, train_sent, train_y),
            (val_X, val_sent, val_y)
        )
        
        # 4. Train baseline models
        baseline_result = self.train_baseline_models(sequences, sentiment_seqs, targets)
        
        # 5. Evaluate all models
        evaluation_results = self.evaluate_all_models(
            (test_X, test_sent, test_y),
            lstm_result,
            baseline_result
        )
        
        # 6. Create visualizations
        self.create_comparison_plots(evaluation_results)
        
        # 7. Save results
        self.save_results(evaluation_results)
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {self.exp_dir}")

if __name__ == "__main__":
    # Run the full experiment
    runner = ExperimentRunner()
    runner.run_full_experiment()