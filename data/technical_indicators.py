#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical Indicators Calculator

纯技术分析指标计算器，只基于OHLCV数据计算各种技术指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import talib


class TechnicalIndicators:
    """
    技术指标计算器
    """
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        输入: DataFrame with columns [Open, High, Low, Close, Volume]
        输出: DataFrame with all technical indicators
        """
        result_df = df.copy()
        
        # 确保列名标准化
        if 'Open' not in result_df.columns and 'open' in result_df.columns:
            result_df = result_df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            })
        
        # 提取OHLCV数据
        open_prices = result_df['Open'].values
        high_prices = result_df['High'].values  
        low_prices = result_df['Low'].values
        close_prices = result_df['Close'].values
        volume = result_df['Volume'].values
        
        # 1. 基础价格特征
        result_df = self._add_basic_features(result_df, close_prices)
        
        # 2. 趋势指标
        result_df = self._add_trend_indicators(result_df, open_prices, high_prices, 
                                             low_prices, close_prices)
        
        # 3. 动量指标
        result_df = self._add_momentum_indicators(result_df, high_prices, low_prices, close_prices)
        
        # 4. 波动率指标
        result_df = self._add_volatility_indicators(result_df, high_prices, low_prices, close_prices)
        
        # 5. 成交量指标
        result_df = self._add_volume_indicators(result_df, close_prices, volume)
        
        # 6. 价格形态特征
        result_df = self._add_pattern_features(result_df, open_prices, high_prices, 
                                             low_prices, close_prices)
        
        # 填充NaN值
        result_df = self._handle_missing_values(result_df)
        
        return result_df
    
    def _add_basic_features(self, df: pd.DataFrame, close_prices: np.ndarray) -> pd.DataFrame:
        """基础价格特征"""
        
        # 收益率
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 价格动量 (不同时间窗口的收益率)
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # 价格位置 (当前价格在最近N天的位置)
        for period in [14, 20]:
            rolling_min = df['Close'].rolling(window=period).min()
            rolling_max = df['Close'].rolling(window=period).max()
            df[f'price_position_{period}'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame, open_prices: np.ndarray, 
                            high_prices: np.ndarray, low_prices: np.ndarray, 
                            close_prices: np.ndarray) -> pd.DataFrame:
        """趋势指标"""
        
        # 移动平均线
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
            
            # 价格相对于均线的位置
            df[f'close_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
            df[f'close_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
        
        # 移动平均线之间的关系
        df['sma_5_20_ratio'] = df['sma_5'] / df['sma_20']
        df['ema_5_20_ratio'] = df['ema_5'] / df['ema_20']
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close_prices)
        
        # 抛物线SAR
        df['sar'] = talib.SAR(high_prices, low_prices)
        
        # ADX (平均趋向指数)
        df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame, high_prices: np.ndarray,
                               low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """动量指标"""
        
        # RSI
        for period in [9, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
        
        # 随机指标
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices)
        df['stochf_k'], df['stochf_d'] = talib.STOCHF(high_prices, low_prices, close_prices)
        
        # Williams %R
        for period in [14, 21]:
            df[f'willr_{period}'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
        
        # CCI (商品通道指数)
        df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        
        # ROC (变动率)
        for period in [10, 20]:
            df[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)
        
        # Ultimate Oscillator
        df['ultosc'] = talib.ULTOSC(high_prices, low_prices, close_prices)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame, high_prices: np.ndarray,
                                 low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """波动率指标"""
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close_prices, timeperiod=20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (真实波动率)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
        
        # 历史波动率
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(252)
        
        # True Range
        df['trange'] = talib.TRANGE(high_prices, low_prices, close_prices)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame, close_prices: np.ndarray, 
                             volume: np.ndarray) -> pd.DataFrame:
        """成交量指标"""
        
        # 成交量移动平均
        for period in [5, 10, 20]:
            df[f'vol_sma_{period}'] = talib.SMA(volume.astype(float), timeperiod=period)
            df[f'vol_ratio_{period}'] = df['Volume'] / df[f'vol_sma_{period}']
        
        # OBV (能量潮)
        df['obv'] = talib.OBV(close_prices, volume.astype(float))
        
        # 资金流量指数 (MFI)
        df['mfi'] = talib.MFI(df['High'].values, df['Low'].values, 
                             close_prices, volume.astype(float), timeperiod=14)
        
        # 量价关系
        df['price_volume'] = df['Close'] * df['Volume']
        df['volume_price_ratio'] = df['Volume'] / df['Close']
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame, open_prices: np.ndarray,
                            high_prices: np.ndarray, low_prices: np.ndarray,
                            close_prices: np.ndarray) -> pd.DataFrame:
        """价格形态特征"""
        
        # K线形态
        df['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        df['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        df['hanging_man'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
        df['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        
        # 缺口
        df['gap_up'] = (df['Open'] > df['High'].shift(1)).astype(int)
        df['gap_down'] = (df['Open'] < df['Low'].shift(1)).astype(int)
        
        # 实体和影线
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['total_range'] = df['High'] - df['Low']
        
        # 相对大小
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        
        # 前向填充
        df = df.fillna(method='ffill')
        
        # 剩余的NaN用0填充（主要是最开始的几行）
        df = df.fillna(0)
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """获取所有技术指标特征列表"""
        
        basic_features = ['returns', 'log_returns']
        basic_features.extend([f'momentum_{p}' for p in [3, 5, 10, 20]])
        basic_features.extend([f'price_position_{p}' for p in [14, 20]])
        
        trend_features = []
        trend_features.extend([f'sma_{p}' for p in [5, 10, 20, 50]])
        trend_features.extend([f'ema_{p}' for p in [5, 10, 20, 50]])
        trend_features.extend([f'close_sma_{p}_ratio' for p in [5, 10, 20, 50]])
        trend_features.extend([f'close_ema_{p}_ratio' for p in [5, 10, 20, 50]])
        trend_features.extend(['sma_5_20_ratio', 'ema_5_20_ratio'])
        trend_features.extend(['macd', 'macd_signal', 'macd_hist', 'sar', 'adx', 'plus_di', 'minus_di'])
        
        momentum_features = []
        momentum_features.extend([f'rsi_{p}' for p in [9, 14, 21]])
        momentum_features.extend(['stoch_k', 'stoch_d', 'stochf_k', 'stochf_d'])
        momentum_features.extend([f'willr_{p}' for p in [14, 21]])
        momentum_features.extend(['cci'])
        momentum_features.extend([f'roc_{p}' for p in [10, 20]])
        momentum_features.append('ultosc')
        
        volatility_features = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position']
        volatility_features.extend([f'atr_{p}' for p in [7, 14, 21]])
        volatility_features.extend([f'volatility_{p}' for p in [10, 20, 30]])
        volatility_features.append('trange')
        
        volume_features = []
        volume_features.extend([f'vol_sma_{p}' for p in [5, 10, 20]])
        volume_features.extend([f'vol_ratio_{p}' for p in [5, 10, 20]])
        volume_features.extend(['obv', 'mfi', 'price_volume', 'volume_price_ratio'])
        
        pattern_features = ['doji', 'hammer', 'hanging_man', 'shooting_star', 'engulfing']
        pattern_features.extend(['gap_up', 'gap_down'])
        pattern_features.extend(['body_size', 'upper_shadow', 'lower_shadow', 'total_range'])
        pattern_features.extend(['body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio'])
        
        all_features = (basic_features + trend_features + momentum_features + 
                       volatility_features + volume_features + pattern_features)
        
        return all_features


def calculate_technical_indicators_for_multiple_stocks(price_files: List[str]) -> pd.DataFrame:
    """
    为多只股票计算技术指标
    """
    calculator = TechnicalIndicators()
    all_data = []
    
    for file_path in price_files:
        print(f"Processing {file_path}...")
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 提取股票代码
        if 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0]
        else:
            # 从文件名提取
            symbol = file_path.split('/')[-1].replace('_price_data.csv', '').replace('.csv', '')
            df['symbol'] = symbol
        
        # 确保Date列存在并转换为datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # 排序确保时间顺序
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 计算技术指标
        df_with_indicators = calculator.calculate_all_indicators(df)
        
        all_data.append(df_with_indicators)
        print(f"  Added {len(df_with_indicators)} records for {symbol}")
    
    # 合并所有股票数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nTotal combined data: {len(combined_df)} records")
    print(f"Symbols: {combined_df['symbol'].unique()}")
    print(f"Features: {len(combined_df.columns)} columns")
    
    return combined_df


if __name__ == "__main__":
    # 测试技术指标计算
    price_files = [
        'AAPL_price_data.csv',
        'GOOGL_price_data.csv', 
        'MSFT_price_data.csv',
        'NVDA_price_data.csv',
        'TSLA_price_data.csv'
    ]
    
    # 计算技术指标
    technical_data = calculate_technical_indicators_for_multiple_stocks(price_files)
    
    # 保存结果
    technical_data.to_csv('data/processed/technical_indicators_data.csv', index=False)
    print(f"Technical indicators saved to data/processed/technical_indicators_data.csv")
    
    # 显示特征列表
    calculator = TechnicalIndicators()
    feature_list = calculator.get_feature_list()
    print(f"\nTotal technical features: {len(feature_list)}")
    print("Feature categories:")
    print("- Basic features:", len([f for f in feature_list if any(x in f for x in ['returns', 'momentum', 'position'])]))
    print("- Trend features:", len([f for f in feature_list if any(x in f for x in ['sma', 'ema', 'macd', 'sar', 'adx'])]))
    print("- Momentum features:", len([f for f in feature_list if any(x in f for x in ['rsi', 'stoch', 'willr', 'cci', 'roc', 'ultosc'])]))
    print("- Volatility features:", len([f for f in feature_list if any(x in f for x in ['bb_', 'atr_', 'volatility_', 'trange'])]))
    print("- Volume features:", len([f for f in feature_list if any(x in f for x in ['vol_', 'obv', 'mfi', 'volume'])]))
    print("- Pattern features:", len([f for f in feature_list if any(x in f for x in ['doji', 'hammer', 'gap', 'body', 'shadow'])]))