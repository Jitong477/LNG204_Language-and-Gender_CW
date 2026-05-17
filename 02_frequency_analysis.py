import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from collections import Counter

# ==================== 路径与设置 ====================
BASE_DIR = "/Users/zoujitong/PyCharmMiscProject/莓辣"
INPUT_CSV = f"{BASE_DIR}/output/corpus/full.csv"
STOPWORDS_PATH = f"{BASE_DIR}/Data/stopwords.txt"  # 停用词路径
OUTPUT_DIR = f"{BASE_DIR}/output/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体 (Mac)
FONT_PATH = '/System/Library/Fonts/STHeiti Medium.ttc'


# ==================== 加载停用词 ====================
def load_stopwords(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f if line.strip()])
    else:
        print("⚠️ 未找到停用词文件，将只使用默认过滤")
        return set()


stopwords = load_stopwords(STOPWORDS_PATH)
# 你也可以在这里手动增加几个恼人的词
stopwords.update(['我们', '自己', '这个', '一个', '没有', '感觉', '觉得'])

# ==================== 读取数据 ====================
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

# ==================== 绘图准备 ====================
periods = ['2016-2019', '2020-2022', '2023-2026']
period_labels = [
    'Period 1: 2016–2019\n(Institutional Phase)',
    'Period 2: 2020–2022\n(Transitional Phase)',
    'Period 3: 2023–2026\n(Subjectivity Phase)'
]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# ==================== 循环生成标准化词云 ====================
for i, (period, label) in enumerate(zip(periods, period_labels)):
    # 提取该时期所有分词
    subset = df[df['period'] == period]['seg_text'].dropna()
    all_tokens = ' '.join(subset).split()
    total_tokens = len(all_tokens)

    # 计算词频并进行标准化 (per 1,000 tokens)
    counts = Counter(all_tokens)

    # 过滤单字词和停用词
    rel_freqs = {
        word: (count / total_tokens) * 1000
        for word, count in counts.items()
        if word not in stopwords and len(word) > 1
    }

    # 生成词云
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800,
        height=700,
        background_color='white',
        max_words=100,
        colormap='plasma',
        collocations=False
    ).generate_from_frequencies(rel_freqs)

    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(label, fontsize=15, fontweight='bold', pad=15)

# ==================== 保存与显示 ====================
plt.suptitle('Figure 1. Normalized Word Clouds of MAYLOVE Corpus (Normalized per 1,000 tokens)',
             fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig01_wordcloud_periods.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 成功！标准化词云图已保存至: {OUTPUT_DIR}/fig01_wordcloud_periods.png")
