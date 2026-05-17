"""
Step 2: 正式清洗 + 分词 + 分时期
输入：原始CSV（莓辣MAYLOVE.csv）
输出：output/corpus/ 下的CSV文件（含seg_text列）
"""

import pandas as pd
import jieba
import os
import re
from datetime import datetime

# ==================== 路径配置 ====================
BASE_DIR = "/Users/zoujitong/PyCharmMiscProject/莓辣"
INPUT_CSV = os.path.join(BASE_DIR, "莓辣MAYLOVE.csv")
STOPWORDS_FILE = os.path.join(BASE_DIR, "Data/stopwords.txt")
USER_DICT = os.path.join(BASE_DIR, "Data/mydict.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output/corpus")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 加载停用词 ====================
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f if line.strip()])
    print(f"加载停用词: {len(stopwords)} 个")
    return stopwords

# ==================== 加载自定义词典 ====================
def load_userdict(filepath):
    jieba.load_userdict(filepath)
    print("自定义词典加载完成")

# ==================== 文本清洗与分词 ====================
def clean_and_cut(text, stopwords):
    if not isinstance(text, str):
        return ""
    # 只保留中文、英文、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    words = jieba.cut(text)
    # 过滤停用词和单字词（长度>1）
    filtered = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(filtered)

# ==================== 分配时间段 ====================
def assign_period(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        year = dt.year
    except:
        return "unknown"
    if 2016 <= year <= 2019:
        return "2016-2019"
    elif 2020 <= year <= 2022:
        return "2020-2022"
    elif 2023 <= year <= 2026:
        return "2023-2026"
    else:
        return "other"

# ==================== 主函数 ====================
def main():
    print("=" * 50)
    print("开始正式清洗 + 分词 + 分时期")
    print("=" * 50)

    # 1. 加载停用词和词典
    stopwords = load_stopwords(STOPWORDS_FILE)
    load_userdict(USER_DICT)

    # 2. 读取原始CSV
    print("\n正在读取原始数据...")
    with open(INPUT_CSV, 'r', encoding='gb18030', errors='ignore') as f:
        df = pd.read_csv(f)
    print(f"原始数据: {len(df)} 行")

    # 3. 确定列名
    title_col = '文章标题'
    time_col = '发布时间↓'
    content_col = '文章内容'

    # 4. 清洗：去重
    df = df.drop_duplicates(subset=[content_col])
    print(f"去重后: {len(df)} 行")

    # 5. 清洗：去广告/通知
    junk_keywords = '福利|抽奖|通知|停更|公告|课程|购买|加群|训练营|招募'
    df = df[~df[title_col].str.contains(junk_keywords, na=False)]
    print(f"去广告后: {len(df)} 行")

    # 6. 清洗：过滤短文章（<100字）
    df['content_len'] = df[content_col].astype(str).apply(len)
    df = df[df['content_len'] >= 100]
    print(f"过滤短文章后: {len(df)} 行（最终语料库）")

    # 7. 分配时期
    df['period'] = df[time_col].apply(assign_period)

    # 8. 分词
    print("\n正在分词（可能需要几分钟）...")
    df['seg_text'] = df[content_col].apply(lambda x: clean_and_cut(x, stopwords))

    # 9. 过滤分词后为空的行
    df = df[df['seg_text'].str.strip() != '']
    print(f"分词后有效: {len(df)} 行")

    # 10. 保存全量CSV
    full_csv = os.path.join(OUTPUT_DIR, "full.csv")
    df.to_csv(full_csv, index=False, encoding='utf-8-sig')
    print(f"\n全量语料已保存: {full_csv}")

    # 11. 按时期保存
    periods = ['2016-2019', '2020-2022', '2023-2026']
    for period in periods:
        sub_df = df[df['period'] == period].copy()
        if len(sub_df) > 0:
            sub_csv = os.path.join(OUTPUT_DIR, f"{period}.csv")
            sub_df.to_csv(sub_csv, index=False, encoding='utf-8-sig')
            print(f"{period}: {len(sub_df)} 篇 -> {sub_csv}")

    # 12. 统计报告
    print("\n" + "=" * 50)
    print("最终统计：")
    print(f"总文章数: {len(df)}")
    for period in periods:
        cnt = len(df[df['period'] == period])
        print(f"  {period}: {cnt} 篇")
    print("=" * 50)
    print("✅ 完成！")

if __name__ == "__main__":
    main()