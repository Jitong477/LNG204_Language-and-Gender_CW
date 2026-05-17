import pandas as pd
from collections import Counter

# 1. 设置路径（请确保路径与你之前的代码一致）
BASE_DIR = "/Users/zoujitong/PyCharmMiscProject/莓辣"
INPUT_CSV = f"{BASE_DIR}/output/corpus/full.csv"

# 2. 读取数据
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

# 3. 定义我们要对比的性别词组
gender_pairs = [('她们', '他们'), ('女性', '男性')]
periods = ['2016-2019', '2020-2022', '2023-2026']

print("=" * 50)
print("MAYLOVE Gender Ratio Analysis (Ratio = Female / Male)")
print("=" * 50)

# 4. 开始统计
results = []
for period in periods:
    print(f"\n【Period: {period}】")
    # 提取该时期分词
    subset = df[df['period'] == period]['seg_text'].dropna()
    all_tokens = ' '.join(subset).split()
    counter = Counter(all_tokens)

    for f_word, m_word in gender_pairs:
        f_count = counter.get(f_word, 0)
        m_count = counter.get(m_word, 0)

        # 计算比例
        ratio = f_count / m_count if m_count > 0 else f_count
        print(f"  {f_word}({f_count}) vs {m_word}({m_count}) -> Ratio: {ratio:.2f}")

        results.append({
            'Period': period,
            'Pair': f"{f_word}/{m_word}",
            'Female_Count': f_count,
            'Male_Count': m_count,
            'Ratio': round(ratio, 2)
        })

print("\n" + "=" * 50)
print("如果 Ratio 显著上升，说明女性主体性在话语中占据了绝对主导。")