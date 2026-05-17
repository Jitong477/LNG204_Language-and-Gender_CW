import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # --- 1. 读取与预处理 ---
    df = pd.read_csv("/Users/zoujitong/PyCharmMiscProject/莓辣/output/corpus/full.csv")
    df.columns = df.columns.str.strip()
    texts = [text.split() for text in df['seg_text'].dropna()]

    dictionary = corpora.Dictionary(texts)
    # 这里的 no_below=5 和 no_above=0.5 就是调参的一部分，控制词频过滤边界
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print(f"字典大小: {len(dictionary)}")
    print(f"语料文档数: {len(corpus)}")

    # --- 2. 核心调参循环 (测试 K = 3, 4, 5, 6) ---
    k_values = [3, 4, 5, 6]
    coherence_scores = []
    models_dict = {}

    print("\n--- 开始计算不同K值下模型的一致性得分 ---")
    for k in k_values:
        print(f"正在训练 K={k} 的 LDA 模型...")
        # 调参点 1: num_topics 动态变化
        # 调参点 2: passes=30（保持足够迭代），random_state=42（保证可重复）
        model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=30,
            random_state=42
        )
        models_dict[k] = model

        # 计算当前K值下的 Coherence Score
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        score = coherence_model.get_coherence()
        coherence_scores.append(score)
        print(f"-> K={k} 的 Coherence (C_v) 为: {score:.4f}\n")

    # --- 3. 自动绘制并保存 Coherence 曲线图 ---
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, coherence_scores, marker='o', color='b', linewidth=2)
    plt.title('Coherence Score (C_v) by Topic Number (K)', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Topics (K)', fontsize=10)
    plt.ylabel('Coherence Score', fontsize=10)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs("/Users/zoujitong/PyCharmMiscProject/莓辣/output/figures", exist_ok=True)
    fig_path = "/Users/zoujitong/PyCharmMiscProject/莓辣/output/figures/lda_coherence_curve.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Coherence 曲线图已保存至: {fig_path}")
    plt.show()

    # --- 4. 打印每个模型的关键词，方便你做最终的选择 ---
    print("\n--- 各个K值模型的主题关键词对照 ---")
    for k in k_values:
        print(f"\n================ 💡 当 K = {k} 时 ================")
        for i, topic in models_dict[k].print_topics(num_words=10):
            print(f"主题 {i + 1}: {topic}")

    # --- 5. 保存你最终满意的模型 (这里先默认保存你原本想要的K=5，可根据运行结果修改) ---
    # 比如你运行后发现 K=4 最好，就把下面改成 models_dict[4]
    final_k = 5
    os.makedirs("/Users/zoujitong/PyCharmMiscProject/莓辣/output/lda", exist_ok=True)
    models_dict[final_k].save(f"/Users/zoujitong/PyCharmMiscProject/莓辣/output/lda/lda_k{final_k}")
    dictionary.save("/Users/zoujitong/PyCharmMiscProject/莓辣/output/lda/dictionary")
    print(f"\n✅ 最终选择的 K={final_k} 模型已保存。")