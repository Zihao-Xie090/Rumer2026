# 可解释的谣言检测

本项目用于《人工智能导论》大作业：给定一条推文文本，输出二分类检测结果，并生成一段可解释的判断依据。

当前实现采用 `TF-IDF + Logistic Regression` 作为 baseline。模型根据训练集中词语、短语与标签之间的统计关系完成分类，并用对当前预测贡献最大的关键词生成解释。

## 目录结构

```text
rumer2026/
  train.csv
  val.csv
  README.md
  requirements.txt
  .gitignore
  .gitattributes
  src/
    config.py
    data_utils.py
    train.py
    evaluate.py
    predict.py
    explain.py
  models/
    .gitkeep
  outputs/
    .gitkeep
  docs/
    report.md
```

## 环境安装

建议使用 Python 3.10 或 Python 3.11。

在项目根目录执行：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果你的命令行里 `python` 不可用，需要先安装 Python，或安装 Miniconda 后再创建环境。

## 数据说明

数据文件位于项目根目录：

- `train.csv`：训练集
- `val.csv`：验证集

字段含义：

- `id`：推文编号
- `text`：推文文本
- `label`：标签，`0` 表示非谣言，`1` 表示谣言
- `event`：事件编号，可用于数据分析

当前模型只使用 `text` 作为输入特征。

## 训练模型

在项目根目录执行：

```bash
python -m src.train
```

训练完成后，模型会保存到：

```text
models/tfidf_lr.joblib
```

## 评估模型

```bash
python -m src.evaluate
```

评估结果会保存到：

```text
outputs/metrics.json
outputs/val_predictions.csv
```

其中 `val_predictions.csv` 包含每条验证样本的真实标签、预测标签、预测置信度和解释文本。

## 单条预测

```bash
python -m src.predict --text "BREAKING: police just announced new details about the case"
```

输出包括：

- `prediction`：预测标签
- `label_name`：标签含义
- `probability`：模型置信度
- `top_features`：对判断贡献较大的关键词
- `explanation`：自然语言判断依据

## 方法简介

1. 文本预处理：将 URL 替换为 `URL`，将用户提及替换为 `USER`，小写化并清理多余空格。
2. 特征提取：使用 unigram 和 bigram 的 TF-IDF 特征。
3. 分类模型：使用带类别权重的逻辑回归，缓解类别数量不完全均衡的问题。
4. 解释生成：根据当前文本中 TF-IDF 特征与模型权重的乘积，提取对预测贡献最大的词语或短语，并生成解释。

## GitHub 提交建议

不要在 `C:\Users\36503` 目录执行 `git add .`。

请始终进入项目目录后再操作：

```bash
cd /d C:\Users\36503\Desktop\rumer2026
git status
git add .
git commit -m "Add baseline rumor detection model"
git push
```

如果误把仓库建到了用户主目录，需要删除 `C:\Users\36503\.git` 后重新初始化项目目录。
