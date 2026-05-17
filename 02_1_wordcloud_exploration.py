import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from collections import Counter

df = pd.read_csv("/Users/zoujitong/PyCharmMiscProject/莓辣/output/corpus/full.csv")
df.columns = df.columns.str.strip()

os.makedirs("/Users/zoujitong/PyCharmMiscProject/莓辣/output/figures", exist_ok=True)

# 额外停用词
extra_stopwords = {'我们', '自己', '一个', '没有', '可以', '这个', '什么', '知道', '为什么', '不要', '一样', '应该',
                   '发现', '觉得', '时间', '不同', '问题', '成为', '影响', '存在', '正常', '发生','表情', '方式', '过程',
                   '只有', '最后', '告诉', '使用', '开始', '已经', '刘强'
}

periods = ['2016-2019', '2020-2022', '2023-2026']
period_labels = [
    'Period 1: 2016–2019\n(Institutional Phase)',
    'Period 2: 2020–2022\n(Transitional Phase)',
    'Period 3: 2023–2026\n(Subjectivity Phase)'
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (period, label) in enumerate(zip(periods, period_labels)):
    subset = df[df['period'] == period]['seg_text'].dropna()
    all_tokens = ' '.join(subset).split()
    total_tokens = len(all_tokens)  # 该时期的总词数

    # 计算相对频率 (比如每千词出现次数)
    counts = Counter(all_tokens)
    rel_freqs = {word: (count / total_tokens) * 1000 for word, count in counts.items()
                 if word not in extra_stopwords}

    wc = WordCloud(
        font_path='/System/Library/Fonts/STHeiti Medium.ttc',
        width=600, height=500,
        background_color='white',
        max_words=80
    ).generate_from_frequencies(rel_freqs)  # 使用相对频率生成

    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(label, fontsize=13, fontweight='bold', pad=10)

plt.suptitle('Figure 1. Word Clouds of MAYLOVE Corpus by Period (N = 1,865)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/zoujitong/PyCharmMiscProject/莓辣/output/figures/fig01_wordcloud_periods.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("saved")