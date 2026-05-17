import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from collections import Counter

# --- 1. 路径配置 ---
INPUT_CSV = "/Users/zoujitong/PyCharmMiscProject/莓辣/output/corpus/full.csv"
STOPWORDS_PATH = "/Users/zoujitong/PyCharmMiscProject/莓辣/Data/stopwords.txt"
OUTPUT_FIG = "/Users/zoujitong/PyCharmMiscProject/莓辣/output/figures/fig03_wordcloud_all.png"
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"

# --- 2. 加载数据与停用词 ---
print("正在加载数据...")
df = pd.read_csv(INPUT_CSV)
texts = df['seg_text'].dropna().astype(str).tolist()

# 加载基础停用词表
stopwords = set()
if os.path.exists(STOPWORDS_PATH):
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f.readlines()])

# 补充在之前代码中发现的额外停用词
extra_stopwords = {'我们', '自己', '一个', '没有', '可以', '这个', '什么', '知道', '为什么', '不要', '一样', '应该',
                   '发现', '觉得', '时间', '不同', '问题', '成为', '影响', '存在', '正常', '发生','表情', '方式', '过程',
                   '只有', '最后', '告诉', '使用', '开始', '已经', '刘强'}
stopwords.update(extra_stopwords)

# --- 3. 词频统计与清洗 ---
print("正在统计词频...")
all_tokens = ' '.join(texts).split()
total_tokens = len(all_tokens)

counts = Counter(all_tokens)
# 使用相对频率 (每千词频) 以保证统计上的严谨性
rel_freqs = {word: (count / total_tokens) * 1000 for word, count in counts.items()
             if word not in stopwords and len(word) > 1}

# --- 4. 生成整体词云图 ---
print("正在生成词云...")
wc = WordCloud(
    font_path=FONT_PATH,
    width=1200,
    height=800,
    background_color='white',
    max_words=150,
    colormap='viridis',
    prefer_horizontal=0.8
).generate_from_frequencies(rel_freqs)

# --- 5. 绘图与保存 ---
plt.figure(figsize=(15, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Figure 3. Aggregate Word Cloud of MAYLOVE Corpus (2016–2026)',
          fontsize=16, fontweight='bold', pad=20)

os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
print(f"✅ 整体词云图已保存至: {OUTPUT_FIG}")
plt.show()