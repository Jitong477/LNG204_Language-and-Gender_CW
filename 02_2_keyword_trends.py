import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from collections import Counter

# 设置中文字体
font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

df = pd.read_csv("/Users/zoujitong/PyCharmMiscProject/莓辣/output/corpus/full.csv")
df.columns = df.columns.str.strip()
df['发布时间↓'] = pd.to_datetime(df['发布时间↓'])
df['year'] = df['发布时间↓'].dt.year

group1 = {'讲座': 'lecture/workshop', '性教育': 'sexuality education'}
group2 = {'她们': 'they (female)', '感受': 'feeling/experience', '选择': 'choice/agency'}
group3 = {'身体': 'body', '关系': 'relationship', '伴侣': 'partner'}

all_words = {**group1, **group2, **group3}

results = []
for year, group in df.groupby('year'):
    all_tokens = ' '.join(group['seg_text'].dropna()).split()
    total = len(all_tokens)
    counter = Counter(all_tokens)
    for word, label in all_words.items():
        freq = counter.get(word, 0) / total * 1000
        results.append({'year': year, 'word': word, 'label': label, 'freq_per_1000': freq})

result_df = pd.DataFrame(results)

# 竖排三张
fig, axes = plt.subplots(3, 1, figsize=(10, 14))

groups = [
    (group1, ['#2d6a4f', '#52b788'], 'Institutional Discourse'),
    (group2, ['#7b2d8b', '#c77dff', '#e0aaff'], 'Subjectivity Discourse'),
    (group3, ['#1d3557', '#457b9d', '#a8dadc'], 'Relational Discourse'),
]

for ax, (group, cols, title) in zip(axes, groups):
    for (word, label), color in zip(group.items(), cols):
        subset = result_df[result_df['word'] == word]
        ax.plot(subset['year'], subset['freq_per_1000'],
                label=f'{word} ({label})',
                color=color, linewidth=2.5, marker='o', markersize=5)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Frequency per 1,000 tokens', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=2019.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=2022.5, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Figure 2. Keyword Frequency Trends in MAYLOVE Corpus (2016–2026)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/zoujitong/PyCharmMiscProject/莓辣/output/figures/fig02_keyword_trends.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("saved")