#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import yfinance as yf
import pandas as pd
import numpy as np
import yaml


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_utc_series(idx) -> pd.Series:
    """
    将 DatetimeIndex 或任意可解析的时间列安全转换为 pandas datetime（UTC tz-aware）
    """
    dt = pd.to_datetime(idx, errors="coerce", utc=True)
    return dt


class PriceCollector:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}

    def get_stock_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        获取股票价格数据，并计算常用技术指标。返回每个 symbol 的 DataFrame（含 Date 列、symbol 列）
        """
        stock_data: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                # yfinance 返回的 index 通常是日期索引；统一转为 UTC tz-aware 后落到列
                date_utc = _to_utc_series(data.index)
                data = data.copy()
                data.insert(0, "Date", date_utc)

                # 计算技术指标
                data["Returns"] = data["Close"].pct_change()
                data["MA_5"] = data["Close"].rolling(window=5, min_periods=1).mean()
                data["MA_20"] = data["Close"].rolling(window=20, min_periods=1).mean()
                data["Volatility"] = data["Returns"].rolling(window=20, min_periods=1).std()

                # 未来一步价格方向与变化（注意：最后一行会是 NaN）
                data["Price_Direction"] = (data["Close"].shift(-1) > data["Close"]).astype("Int64")
                data["Price_Change"] = data["Close"].shift(-1) - data["Close"]
                data["Price_Change_Pct"] = data["Price_Change"] / data["Close"]

                # 加上 symbol 列
                data["symbol"] = symbol

                # 统一列顺序（若某些列不存在也不会报错）
                preferred = [
                    "Date", "symbol",
                    "Open", "High", "Low", "Close", "Adj Close", "Volume",
                    "Returns", "MA_5", "MA_20", "Volatility",
                    "Price_Direction", "Price_Change", "Price_Change_Pct",
                ]
                cols = [c for c in preferred if c in data.columns] + [c for c in data.columns if c not in preferred]
                data = data[cols]

                stock_data[symbol] = data
                print(f"[OK] collected {symbol}: {len(data)} rows")

            except Exception as e:
                print(f"[WARN] Error collecting data for {symbol}: {e}")

        return stock_data

    def get_market_overview(self, symbols: List[str], date_range: int = 30) -> pd.DataFrame:
        """
        获取市场概览（近 date_range 天的粗略波动/涨跌/成交均量等）
        """
        end_date = datetime.now(tz=timezone.utc)
        start_date = end_date - timedelta(days=int(date_range))

        overview_rows = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if len(data) == 0:
                    continue
                dt_utc = _to_utc_series(data.index)
                close = data["Close"].astype(float)
                latest_price = float(close.iloc[-1])
                price_change = float((latest_price - close.iloc[0]) / close.iloc[0])
                volatility = float(close.pct_change().std() or 0.0)
                volume = float(data["Volume"].mean() or 0.0)

                overview_rows.append({
                    "Symbol": symbol,
                    "Latest_Price": latest_price,
                    "Price_Change_Pct": price_change,
                    "Volatility": volatility,
                    "Avg_Volume": volume,
                    "Date": dt_utc.iloc[-1],
                })
            except Exception as e:
                print(f"[WARN] Error getting overview for {symbol}: {e}")

        return pd.DataFrame(overview_rows)

    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "./"):
        """
        保存每个 symbol 的 CSV，并额外**合并**为一个总表 prices.csv（含 symbol 列、Date 列 UTC）
        """
        _ensure_dir(output_dir)

        merged_list = []
        for symbol, df in data.items():
            # 单股文件
            fp_single = os.path.join(output_dir, f"{symbol}_price_data.csv")
            df.to_csv(fp_single, index=False)
            print(f"[OK] saved {symbol} -> {fp_single} ({len(df)} rows)")
            merged_list.append(df)

        # 合并所有
        if merged_list:
            merged = pd.concat(merged_list, ignore_index=True)

            # 保障 Date 列是 UTC tz-aware（以免后续对齐时报 tz 错）
            merged["Date"] = _to_utc_series(merged["Date"])

            # 保存总表
            fp_merged = os.path.join(output_dir, "prices.csv")
            merged.to_csv(fp_merged, index=False)
            print(f"[OK] saved merged prices -> {fp_merged} ({len(merged)} rows)")
        else:
            print("[WARN] nothing to merge; merged prices.csv not written")


def parse_args():
    ap = argparse.ArgumentParser(description="Collect price data and save per-symbol + merged CSV.")
    ap.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    ap.add_argument("--symbols", type=str, default="", help="Comma-separated tickers, e.g. 'AAPL,MSFT'. Overrides config.")
    ap.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD. Overrides config.")
    ap.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD. Overrides config.")
    ap.add_argument("--outdir", type=str, default="./", help="Output directory")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pc = PriceCollector(config_path=args.config)

    # symbols: CLI > config > fallback
    cli_syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    cfg_syms = (pc.config.get("data", {}) or {}).get("symbols", []) if pc.config else []
    symbols = cli_syms or cfg_syms or ["AAPL", "MSFT"]

    # dates: CLI > config > fallback（近一年）
    start = args.start or (pc.config.get("data", {}) or {}).get("start_date", "")
    end = args.end or (pc.config.get("data", {}) or {}).get("end_date", "")
    if not start or not end:
        end_dt = datetime.now(tz=timezone.utc).date()
        start_dt = end_dt - timedelta(days=365)
        start = start or str(start_dt)
        end = end or str(end_dt)

    print(f"Collecting prices for: {symbols} | {start} → {end}")
    stock_data = pc.get_stock_data(symbols, start, end)

    pc.save_data(stock_data, output_dir=args.outdir)

    # 可选：打印一个市场概览
    overview = pc.get_market_overview(symbols, date_range=30)
    if len(overview):
        print("\nMarket Overview (last ~30 days):")
        print(overview.head(10).to_string(index=False))
