"""
Step 1: 原始数据勘探
目的：识别噪声、暗号、需要合并的专业词
"""

import pandas as pd
import jieba
import re
from collections import Counter

# ==================== 路径 ====================
INPUT_CSV = "/Users/zoujitong/PyCharmMiscProject/莓辣/莓辣MAYLOVE.csv"

# ==================== 读取原始数据 ====================
print("正在读取原始数据...")
with open(INPUT_CSV, 'r', encoding='gb18030', errors='ignore') as f:
    df = pd.read_csv(f)

print(f"原始数据行数: {len(df)}")

# 确认列名
title_col = '文章标题'
content_col = '文章内容'

print(f"使用列: 标题={title_col}, 内容={content_col}")

# ==================== 简单分词函数 ====================
def simple_cut(text):
    """保留中文、英文、数字，分词，只保留长度大于1的词（不含单字）"""
    if not isinstance(text, str):
        return []
    # 只保留中文、英文、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    words = jieba.cut(text)
    # 只保留长度 > 1 的词
    return [w for w in words if len(w) > 1]

# ==================== 统计词频 ====================
print("\n正在统计词频（长度>1）...")
all_words = []
for text in df[content_col]:
    all_words.extend(simple_cut(text))

word_freq = Counter(all_words)
print(f"总词数: {len(all_words)}")
print(f"唯一词数: {len(word_freq)}")

print("\n--- 高频词 Top 50（长度>1）---")
for word, freq in word_freq.most_common(50):
    print(f"{word}: {freq}")

# ==================== 保存结果 ====================
freq_df = pd.DataFrame(word_freq.most_common(200), columns=['word', 'freq'])
freq_df.to_csv("exploration_wordfreq_len2.csv", index=False, encoding='utf-8-sig')

print("\n✅ 勘探完成！")
print("输出文件: exploration_wordfreq_len2.csv")