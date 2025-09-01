#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全自动数据收集脚本
收集价格数据 -> 收集新闻数据 -> 数据对齐 -> 生成LSTM训练序列
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 添加当前目录到Python路径，以便导入其他模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from news_collector import NewsCollector
from price_collector import PriceCollector  
from data_aligner import DataAligner


def main():
    parser = argparse.ArgumentParser(description="一键收集所有数据并对齐")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,TSLA,NVDA", 
                        help="股票代码，逗号分隔")
    parser.add_argument("--days_back", type=int, default=365, 
                        help="收集多少天的数据")
    parser.add_argument("--news_days", type=int, default=30,
                        help="新闻数据天数")
    parser.add_argument("--sequence_length", type=int, default=30,
                        help="LSTM序列长度")
    parser.add_argument("--target", type=str, default="Price_Direction",
                        help="预测目标列")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="输出目录")
    parser.add_argument("--newsapi_key", type=str, default="",
                        help="NewsAPI密钥（可选）")
    
    args = parser.parse_args()
    
    # 解析股票代码
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    
    print("=" * 60)
    print("🚀 开始全自动数据收集流程")
    print(f"📊 股票代码: {symbols}")
    print(f"📅 数据天数: {args.days_back} 天")
    print(f"📰 新闻天数: {args.news_days} 天") 
    print("=" * 60)
    
    # 计算日期范围
    end_date = datetime.now(tz=timezone.utc).date()
    start_date = end_date - timedelta(days=args.days_back)
    
    try:
        # 步骤1: 收集价格数据
        print("\n📈 步骤1: 收集价格数据...")
        price_collector = PriceCollector()
        stock_data = price_collector.get_stock_data(symbols, str(start_date), str(end_date))
        
        if not stock_data:
            print("❌ 未能收集到价格数据，退出")
            return
            
        price_collector.save_data(stock_data, output_dir=args.output_dir)
        print(f"✅ 价格数据收集完成，共 {sum(len(df) for df in stock_data.values())} 条记录")
        
        # 步骤2: 收集新闻数据
        print("\n📰 步骤2: 收集新闻数据...")
        news_collector = NewsCollector(newsapi_key=args.newsapi_key or None)
        news_df = news_collector.get_financial_news(symbols, days_back=args.news_days, sleep_sec=0.5)
        
        if news_df.empty:
            print("⚠️ 未收集到新闻数据，仅使用价格数据")
        else:
            print(f"📄 原始新闻数据: {len(news_df)} 条")
            
        # 清理新闻数据
        news_df = news_collector.clean_news_data(news_df)
        
        # 保存新闻数据
        news_output = os.path.join(args.output_dir, "news_data.csv")
        news_collector.save_news_data(news_df, filename=news_output)
        print(f"✅ 新闻数据处理完成，清理后 {len(news_df)} 条记录")
        
        # 步骤3: 数据对齐和特征工程
        print("\n🔗 步骤3: 数据对齐和特征工程...")
        aligner = DataAligner()
        
        price_path = os.path.join(args.output_dir, "prices.csv")
        news_path = news_output
        
        # 运行完整对齐流程
        aligned_df, X, y = aligner.run_full_alignment(
            price_path=price_path,
            news_path=news_path,
            sequence_length=args.sequence_length,
            target_col=args.target
        )
        
        if aligned_df.empty:
            print("❌ 数据对齐失败")
            return
            
        print(f"✅ 数据对齐完成，合并数据: {aligned_df.shape}")
        
        # 步骤4: 显示结果摘要
        print("\n" + "=" * 60)
        print("🎉 数据收集流程完成！")
        print("=" * 60)
        
        # 文件清单
        files_created = []
        for symbol in symbols:
            symbol_file = os.path.join(args.output_dir, f"{symbol}_price_data.csv")
            if os.path.exists(symbol_file):
                files_created.append(f"{symbol}_price_data.csv")
                
        files_created.extend([
            "prices.csv (合并的价格数据)",
            "news_data.csv (新闻数据)", 
            "aligned_data.csv (对齐后的数据)"
        ])
        
        if len(X) > 0:
            files_created.extend([
                "sequences/X.npy (训练特征序列)",
                "sequences/y.npy (训练目标)",
                "sequences/feature_names.txt (特征名称)"
            ])
            
        print("📁 生成的文件:")
        for f in files_created:
            print(f"  • {f}")
            
        # 数据统计
        print(f"\n📊 数据统计:")
        print(f"  • 股票数量: {len(symbols)}")
        print(f"  • 价格记录: {len(aligned_df)} 条")
        print(f"  • 新闻记录: {len(news_df)} 条")
        if len(X) > 0:
            print(f"  • LSTM序列: {X.shape[0]} 个，每个长度 {X.shape[1]}")
            print(f"  • 特征维度: {X.shape[2]}")
            print(f"  • 目标分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        print(f"\n🎯 下一步: 使用生成的序列数据训练LSTM模型")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 流程执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 确保numpy可用（用于统计输出）
    try:
        import numpy as np
    except ImportError:
        print("警告: numpy未安装，部分统计功能不可用")
        np = None
        
    main()