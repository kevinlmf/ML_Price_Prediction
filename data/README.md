# LLM Price Prediction - 数据收集模块

这个目录包含了用于收集和处理股票价格与新闻数据的完整工具集，专为LSTM训练准备数据。

## 📁 文件结构

```
data/
├── news_collector.py     # 新闻数据收集器 (Yahoo Finance + NewsAPI)
├── price_collector.py    # 价格数据收集器 (Yahoo Finance)
├── data_aligner.py      # 数据对齐和特征工程
├── collect_all.py       # 一键收集所有数据
├── config.yaml          # 配置文件
└── README.md           # 本说明文件
```

## 🚀 快速开始

### 方法1: 一键收集所有数据 (推荐)

```bash
cd data
python collect_all.py --symbols "AAPL,MSFT,GOOGL,TSLA,NVDA"
```

这会自动完成：
1. 收集价格数据
2. 收集新闻数据  
3. 数据对齐和特征工程
4. 生成LSTM训练序列

### 方法2: 分步骤执行

```bash
# 1. 收集价格数据
python price_collector.py --symbols "AAPL,MSFT" --start 2023-01-01 --end 2024-01-01

# 2. 收集新闻数据
python news_collector.py --symbols "AAPL,MSFT" --days_back 30

# 3. 数据对齐
python data_aligner.py --price_path prices.csv --news_path news_data.csv
```

## 📊 生成的数据文件

执行后会生成以下文件：

### 原始数据
- `{SYMBOL}_price_data.csv` - 单个股票的价格数据
- `prices.csv` - 所有股票合并的价格数据
- `news_data.csv` - 清理后的新闻数据

### 处理后数据
- `aligned_data.csv` - 价格和新闻对齐后的数据
- `sequences/X.npy` - LSTM输入序列 (样本数, 时间步, 特征数)
- `sequences/y.npy` - LSTM目标序列
- `sequences/feature_names.txt` - 特征名称列表

## 🔧 配置说明

编辑 `config.yaml` 来自定义：

```yaml
data:
  symbols: ["AAPL", "MSFT", "GOOGL"]  # 股票代码
  news_days_back: 30                   # 新闻数据天数

model:
  sequence_length: 30                  # LSTM序列长度
  target_column: "Price_Direction"     # 预测目标

apis:
  newsapi:
    key: "your-api-key"               # NewsAPI密钥(可选)
```

## 📈 数据特征

### 价格特征
- 基础: `Open`, `High`, `Low`, `Close`, `Volume`
- 技术指标: `Returns`, `MA_5`, `MA_20`, `Volatility`  
- 滞后特征: `Close_lag_1`, `Returns_lag_1`, etc.
- 滚动统计: `Close_roll_mean_5`, `Close_roll_std_10`, etc.
- 目标变量: `Price_Direction` (涨跌), `Price_Change_Pct` (涨跌幅)

### 新闻特征
- `news_count` - 每日新闻数量
- `title` - 新闻标题合并
- `full_text` - 新闻全文合并
- `news_count_roll_sum_7` - 7天滚动新闻数量

### 市场特征
- `market_mean_price` - 当日市场均价
- `market_std_price` - 当日市场价格标准差

### 时间特征  
- `day_of_week`, `month`, `quarter`

## 📋 依赖要求

```bash
pip install pandas numpy yfinance pyyaml scikit-learn

# 可选：NewsAPI支持
pip install newsapi-python
```

## 🔑 API配置

### NewsAPI (可选)
1. 注册 [NewsAPI](https://newsapi.org/) 获取免费密钥
2. 设置环境变量: `export NEWSAPI_KEY=your-key`
3. 或在命令行指定: `--newsapi_key your-key`

### Yahoo Finance
免费使用，无需密钥。

## 💡 使用技巧

1. **数据量控制**: 从少量股票开始测试，确认数据质量后再扩展
2. **时间范围**: 建议至少1年的数据以获得足够的训练样本
3. **新闻数据**: NewsAPI有请求限制，合理设置休眠时间
4. **序列长度**: 30天通常是好的起点，可根据需要调整

## ⚠️ 注意事项

- Yahoo Finance数据为美股交易时间
- 新闻数据时区已统一为UTC
- 周末和节假日可能缺少价格数据，会被自动处理
- 确保有足够磁盘空间存储生成的序列文件

## 🐛 故障排除

1. **ImportError**: 检查依赖是否安装完整
2. **空数据**: 检查股票代码是否正确，日期范围是否合理  
3. **API限制**: NewsAPI每日请求有限制，可分批处理
4. **内存不足**: 减少股票数量或缩短时间范围

## 📞 支持

如遇问题，请检查：
1. 网络连接是否正常
2. 股票代码格式是否正确
3. API密钥是否有效
4. 日期格式是否为 YYYY-MM-DD