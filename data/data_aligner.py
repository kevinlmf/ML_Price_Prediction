#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Aligner (single-factor + sequence-friendly)

功能：
1) 解析新闻时间（published_date 失败则从 summary/content 中的 dict 字符串抽 pubDate/displayTime）
2) 将逐条新闻聚合到 “symbol x day”的日度特征
3) 规整价格数据，计算 next_close / next_return / next_direction
4) 使用 merge_asof(prev_close) 按 symbol+时间对齐“当天新闻”与“最近不晚于新闻时间的收盘价”
5) 提供两种调用方式：
   - 传入 DataFrame：run_full_alignment(price_df=..., news_df=...)
   - 传入路径：run_full_alignment(price_path=..., news_path=...)
6) 支持 sequence_length：
   - sequence_length = 1 → 输出 (N, F)
   - sequence_length > 1 → 输出 (N, L, F)，序列特征，可用于 LSTM/Transformer
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


# ------------------------- 工具 -------------------------

def _to_dt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Expect datetime column '{col}' not found.")
    if not is_datetime(df[col]):
        df = df.copy()
        df[col] = _to_dt_utc(df[col])
    return df

def _prepare_for_asof(df: pd.DataFrame, by_col: str, on_col: str) -> pd.DataFrame:
    df = _ensure_dt(df.copy(), on_col)
    df = df[df[on_col].notna()].copy()
    df = df.sort_values([by_col, on_col], kind="mergesort").reset_index(drop=True)
    return df

def _maybe_extract_time_from_text(x: str):
    if not isinstance(x, str) or "{" not in x or "}" not in x:
        return pd.NaT
    try:
        d = ast.literal_eval(x)
        if isinstance(d, dict):
            for k in ["pubDate", "displayTime", "time", "publishedAt", "published_time"]:
                if k in d:
                    return pd.to_datetime(d[k], errors="coerce", utc=True)
    except Exception:
        pass
    return pd.NaT


# ------------------------- I/O -------------------------

def _read_news(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("items", []))
        return pd.DataFrame(data)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported news file format: {path}")

def _read_prices(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported price file format: {path}")


# ------------------------- 核心 -------------------------

def parse_news_times(news_df: pd.DataFrame,
                     time_cols_priority: List[str] = ("published_date", "publishedAt", "time"),
                     fallback_text_cols: List[str] = ("summary", "content")) -> pd.DataFrame:
    df = news_df.copy()
    if "symbol" not in df.columns:
        raise KeyError("news_df must contain 'symbol' column.")

    news_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    for c in time_cols_priority:
        if c in df.columns:
            t = pd.to_datetime(df[c], errors="coerce", utc=True)
            news_time = news_time.fillna(t)

    for c in fallback_text_cols:
        if c in df.columns:
            t = df[c].apply(_maybe_extract_time_from_text)
            news_time = news_time.fillna(t)

    df["news_time"] = news_time
    df = df[df["news_time"].notna()].copy()

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.nan

    return df

def aggregate_daily_news(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.copy()
    df["date"] = df["news_time"].dt.tz_convert("UTC").dt.date
    grp = df.groupby(["symbol", "date"], as_index=False).agg(
        news_cnt=("sentiment_score", "size"),
        sent_mean=("sentiment_score", "mean"),
        sent_median=("sentiment_score", "median"),
        sent_max=("sentiment_score", "max"),
        sent_min=("sentiment_score", "min"),
        last_news_time=("news_time", "max"),
    )
    grp["asof_time"] = pd.to_datetime(grp["last_news_time"], utc=True)
    return grp

def prepare_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()

    if "timestamp" not in df.columns:
        for alias in ["time", "datetime", "date_time", "ts"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "timestamp"})
                break
        else:
            raise KeyError("prices_df must contain a 'timestamp' column.")

    if "close" not in df.columns:
        for alias in ["Close", "price", "adj_close", "closing_price"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "close"})
                break
        else:
            raise KeyError("prices_df must contain 'close' column.")

    if "symbol" not in df.columns:
        raise KeyError("prices_df must contain 'symbol' column.")

    df = _ensure_dt(df, "timestamp")
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.date
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    last_idx = df.groupby(["symbol", "date"])["timestamp"].idxmax()
    daily_close = df.loc[last_idx, ["symbol", "date", "timestamp", "close"]].copy()
    daily_close = daily_close.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    daily_close = daily_close.rename(columns={"timestamp": "close_time"})
    daily_close["close_time"] = pd.to_datetime(daily_close["close_time"], utc=True)
    return daily_close

def compute_targets(daily_close: pd.DataFrame) -> pd.DataFrame:
    df = daily_close.copy()
    df = df.sort_values(["symbol", "close_time"]).reset_index(drop=True)
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)
    df["next_return"] = (df["next_close"] - df["close"]) / df["close"]
    df["next_direction"] = (df["next_return"] > 0).astype(int)
    return df

def align_prev_close(daily_news: pd.DataFrame,
                     prices_df: pd.DataFrame,
                     tolerance_days: int = 3,
                     *,
                     by_col: str = "symbol",
                     left_on: str = "asof_time",
                     right_on: str = "close_time") -> pd.DataFrame:
    left = _prepare_for_asof(daily_news, by_col, left_on)
    right = _prepare_for_asof(prices_df, by_col, right_on)

    tol = pd.Timedelta(days=tolerance_days)
    merged = pd.merge_asof(
        left,
        right,
        by=by_col,
        left_on=left_on,
        right_on=right_on,
        direction="backward",
        tolerance=tol,
        allow_exact_matches=True,
    )
    return merged

def align_price_news(price_df: pd.DataFrame,
                     news_df: pd.DataFrame,
                     tolerance_days: int = 3) -> pd.DataFrame:
    news_parsed = parse_news_times(news_df)
    daily_news = aggregate_daily_news(news_parsed)
    daily_close = prepare_prices(price_df)
    daily_close_target = compute_targets(daily_close)

    aligned = align_prev_close(
        daily_news=daily_news,
        prices_df=daily_close_target,
        tolerance_days=tolerance_days,
        by_col="symbol",
        left_on="asof_time",
        right_on="close_time",
    )
    aligned = aligned.dropna(subset=["close"]).reset_index(drop=True)
    return aligned

def build_feature_matrix(aligned_df: pd.DataFrame,
                         feature_cols: List[str] = ("sent_mean", "news_cnt", "sent_median", "sent_max", "sent_min"),
                         sequence_length: int = 1
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造特征矩阵 X 和目标 y
    - sequence_length == 1: 返回 (N, F)
    - sequence_length > 1: 返回 (N, L, F) 的序列数据
    """
    df = aligned_df.copy().sort_values(["symbol", "close_time"]).reset_index(drop=True)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    X_list, y_list = [], []

    for sym, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].astype(float).values
        target = g["next_return"].astype(float).values

        if sequence_length == 1:
            X_list.append(feats)
            y_list.append(target)
        else:
            for i in range(len(g) - sequence_length + 1):
                X_list.append(feats[i:i+sequence_length])
                y_list.append(target[i+sequence_length-1])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


# ------------------------- 封装类 -------------------------

@dataclass
class DataAligner:
    tolerance_days: int = 3

    def align_price_news(self,
                         price_df: pd.DataFrame,
                         news_df: pd.DataFrame,
                         tolerance_days: int | None = None) -> pd.DataFrame:
        tol = self.tolerance_days if tolerance_days is None else tolerance_days
        return align_price_news(price_df, news_df, tolerance_days=tol)

    def run_full_alignment(self,
                           price_df: pd.DataFrame | None = None,
                           news_df: pd.DataFrame | None = None,
                           tolerance_days: int | None = None,
                           *,
                           price_path: str | None = None,
                           news_path: str | None = None,
                           sequence_length: int = 1,
                           **kwargs
                           ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if price_df is None:
            if price_path is None:
                raise ValueError("Either price_df or price_path must be provided.")
            price_df = _read_prices(price_path)
        if news_df is None:
            if news_path is None:
                raise ValueError("Either news_df or news_path must be provided.")
            news_df = _read_news(news_path)

        tol = self.tolerance_days if tolerance_days is None else tolerance_days
        aligned_df = align_price_news(price_df, news_df, tolerance_days=tol)

        X, y = build_feature_matrix(aligned_df, sequence_length=sequence_length)
        return aligned_df, X, y


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description="News-Price Data Aligner")
    parser.add_argument("--news", type=str, required=True)
    parser.add_argument("--prices", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--tolerance_days", type=int, default=3)
    parser.add_argument("--sequence_length", type=int, default=1)
    args = parser.parse_args()

    news_df = _read_news(args.news)
    price_df = _read_prices(args.prices)

    aligner = DataAligner(tolerance_days=args.tolerance_days)
    aligned_df, X, y = aligner.run_full_alignment(price_df=price_df,
                                                  news_df=news_df,
                                                  sequence_length=args.sequence_length)

    aligned_df.to_csv(args.out, index=False)
    print(f"[OK] Aligned dataset saved to: {args.out}")
    print(f"[INFO] Shapes -> aligned_df={aligned_df.shape}, X={X.shape}, y={y.shape}")

if __name__ == "__main__":
    main()



