import pandas as pd
from gensim import corpora, models
import os
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 1. 加载数据与定稿的词典、模型
    print("正在加载数据与LDA模型...")
    df = pd.read_csv("/Users/zoujitong/PyCharmMiscProject/莓辣/output/corpus/full.csv")
    df.columns = df.columns.str.strip()
    texts = [text.split() for text in df['seg_text'].dropna()]

    # 加载字典与语料
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 重新训练一个满意的 K=5 模型以确保概率完全对齐
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        passes=30,
        random_state=42
    )

    # 2. 抽取每篇文章的主题概率
    print("正在计算每篇文章的主题概率分布...")
    dominant_topics = []
    topic_percentages = []

    # 动态为 DataFrame 准备 5 个主题的概率列
    for i in range(5):
        df[f'Topic_{i + 1}_Prob'] = 0.0

    doc_idx = 0
    for idx, row in df.iterrows():
        if pd.isna(row['seg_text']):
            dominant_topics.append(None)
            topic_percentages.append(None)
            continue

        bow = corpus[doc_idx]
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)

        doc_topics_sorted = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        dominant_topic_num = doc_topics_sorted[0][0] + 1
        dominant_prob = doc_topics_sorted[0][1]

        dominant_topics.append(dominant_topic_num)
        topic_percentages.append(round(dominant_prob, 4))

        for topic_num, prob in doc_topics:
            df.at[idx, f'Topic_{topic_num + 1}_Prob'] = round(prob, 4)

        doc_idx += 1

    df['Dominant_Topic'] = dominant_topics
    df['Dominant_Topic_Prob'] = topic_percentages

    # 3. 计算各个主题的总占比
    print("\n--- 💡 全量语料中各主题的总体盛行度 (Topic Prevalence) ---")
    topic_counts = df['Dominant_Topic'].value_counts(normalize=True) * 100
    for t_num in range(1, 6):
        percentage = topic_counts.get(t_num, 0.0)
        print(f"主题 {t_num} 的全量文章占比: {percentage:.2f}%")
