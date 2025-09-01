#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import yaml
import json
import argparse
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

try:
    from newsapi import NewsApiClient  # 可选依赖
except Exception:
    NewsApiClient = None


class NewsCollector:
    def __init__(self, config_path: str = "config/config.yaml", newsapi_key: Optional[str] = None):
        # 读取配置（可选）
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            # 没有配置文件也不报错，使用默认
            self.config = {}

        # 初始化 NewsAPI（可选）
        self.newsapi = None
        key = newsapi_key or os.getenv("NEWSAPI_KEY", "").strip()
        if key and NewsApiClient is not None:
            try:
                self.newsapi = NewsApiClient(api_key=key)
            except Exception as e:
                print(f"[WARN] 初始化 NewsAPI 失败：{e}. 将仅使用 Yahoo Finance。")

    # ---------------- Yahoo Finance ----------------
    def _get_yahoo_news(self, symbol: str) -> List[Dict]:
        """
        从 Yahoo Finance 获取新闻
        """
        news_list: List[Dict] = []
        try:
            ticker = yf.Ticker(symbol)
            items = getattr(ticker, "news", None) or []
            for item in items:
                # providerPublishTime 可能不存在或不是数字
                ts = item.get("providerPublishTime", None)
                published_dt = None
                if ts is not None:
                    try:
                        # 以 UTC 处理，避免本地时区误差
                        published_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    except Exception:
                        published_dt = None

                news_list.append({
                    "symbol": symbol or "",
                    "title": item.get("title", "") or "",
                    "summary": item.get("summary", "") or item.get("content", "") or "",
                    "content": item.get("content", "") or "",
                    "url": item.get("link", "") or item.get("url", "") or "",
                    "published_date": published_dt,
                    "publisher": item.get("publisher", "") or "",
                    "source": "yahoo_finance",
                })
        except Exception as e:
            print(f"[WARN] Error getting Yahoo news for {symbol}: {e}")
        return news_list

    # ---------------- NewsAPI（可选） ----------------
    def _get_newsapi_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        使用 NewsAPI 获取新闻（需要 NEWSAPI_KEY）
        """
        news_list: List[Dict] = []
        if self.newsapi is None:
            return news_list

        try:
            # 比较保守的查询，避免过多无关结果
            query = f'"{symbol}" AND (stock OR earnings OR financial OR market)'
            articles = self.newsapi.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                from_param=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                page_size=100,
            )

            for article in articles.get("articles", []):
                published_at = article.get("publishedAt", "")
                try:
                    # NewsAPI 返回 ISO8601，结尾常带 Z
                    published_dt = datetime.fromisoformat(
                        published_at.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except Exception:
                    published_dt = None

                news_list.append({
                    "symbol": symbol or "",
                    "title": article.get("title", "") or "",
                    "summary": article.get("description", "") or "",
                    "content": article.get("content", "") or "",
                    "url": article.get("url", "") or "",
                    "published_date": published_dt,
                    "publisher": (article.get("source") or {}).get("name", "") or "",
                    "source": "newsapi",
                })
        except Exception as e:
            print(f"[WARN] Error getting NewsAPI data for {symbol}: {e}")

        return news_list

    # ---------------- 对外主函数 ----------------
    def get_financial_news(self, symbols: List[str], days_back: int = 30, sleep_sec: float = 1.0) -> pd.DataFrame:
        """
        获取金融新闻数据（Yahoo 必跑，NewsAPI 可选）
        """
        news_data: List[Dict] = []

        end_date = datetime.now(tz=timezone.utc)
        start_date = end_date - timedelta(days=int(days_back))

        for symbol in symbols:
            # 先 Yahoo
            news_data.extend(self._get_yahoo_news(symbol))

            # 再 NewsAPI（如可用）
            if self.newsapi is not None:
                news_data.extend(self._get_newsapi_data(symbol, start_date, end_date))

            # 轻微 sleep，防止限流
            time.sleep(float(sleep_sec))

        # 明确 DataFrame 化
        return pd.DataFrame(news_data)

    def clean_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        清理和预处理新闻数据（安全处理缺失列/类型）
        """
        if news_df is None or getattr(news_df, "empty", True):
            return pd.DataFrame(columns=[
                "symbol", "title", "summary", "content", "url",
                "published_date", "publisher", "source", "full_text"
            ])

        # 需要的列，不存在就补齐
        keep_cols = ["symbol", "title", "summary", "content", "url",
                     "published_date", "publisher", "source"]
        for c in keep_cols:
            if c not in news_df.columns:
                news_df[c] = pd.Series([None] * len(news_df))

        # 去重（用 title + symbol，均先填空串保证健壮）
        dedup_key = news_df["title"].astype("string").fillna("") + "||" + news_df["symbol"].astype("string").fillna("")
        news_df = news_df.loc[~dedup_key.duplicated()].copy()

        # 统一时间类型：将 published_date 尝试转换为 datetime[UTC]
        if not pd.api.types.is_datetime64_any_dtype(news_df["published_date"]):
            news_df["published_date"] = pd.to_datetime(news_df["published_date"], errors="coerce", utc=True)

        # 缺失值填充（逐列）
        for c in ["title", "summary", "content", "url", "publisher", "source", "symbol"]:
            news_df[c] = news_df[c].astype("string").fillna("")

        # 排序（无法解析的时间置后）
        news_df = news_df.sort_values("published_date", ascending=False, na_position="last")

        # 合并文本字段用于情绪分析
        news_df["full_text"] = (news_df["title"] + " " + news_df["summary"]).str.strip()

        # 统一列顺序
        ordered = ["symbol", "title", "summary", "content", "full_text",
                   "url", "publisher", "source", "published_date"]
        return news_df[ordered]

    def save_news_data(self, news_df: pd.DataFrame, filename: str = "news_data.csv"):
        """
        保存新闻数据，自动创建目录
        """
        # 如果filename是绝对路径就直接使用，否则保存到当前目录
        if os.path.isabs(filename):
            fp = Path(filename)
        else:
            fp = Path(filename)
        
        # 确保父目录存在
        fp.parent.mkdir(parents=True, exist_ok=True)

        # 根据扩展名决定保存格式
        if str(fp).lower().endswith(".jsonl"):
            news_df.to_json(fp, orient="records", lines=True, force_ascii=False, date_format="iso")
        else:
            news_df.to_csv(fp, index=False)
        print(f"[OK] Saved news data -> {fp} | rows={len(news_df)}")


def parse_args():
    p = argparse.ArgumentParser(description="Collect and clean financial news.")
    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML.")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols, e.g., 'AAPL,MSFT'. Overrides config.")
    p.add_argument("--days_back", type=int, default=30, help="How many days back to fetch.")
    p.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between symbols.")
    p.add_argument("--out", type=str, default="news_data.csv", help="Output path (.csv or .jsonl).")
    p.add_argument("--newsapi_key", type=str, default="", help="NewsAPI key (optional, overrides ENV NEWSAPI_KEY).")
    return p.parse_args()


def main():
    args = parse_args()

    collector = NewsCollector(config_path=args.config, newsapi_key=args.newsapi_key or None)

    # 解析 symbols：CLI > config > 默认
    cli_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    cfg_symbols = (collector.config.get("data", {}) or {}).get("symbols", []) if collector.config else []
    symbols = cli_symbols or cfg_symbols or ["AAPL", "MSFT"]

    print(f"Collecting financial news for: {symbols} (days_back={args.days_back})")
    news_df = collector.get_financial_news(symbols, days_back=args.days_back, sleep_sec=args.sleep)

    print("Cleaning news...")
    news_df = collector.clean_news_data(news_df)

    collector.save_news_data(news_df, filename=os.path.basename(args.out))


if __name__ == "__main__":
    main()
