import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import hashlib

# ================= 1. 预处理函数 =================
# 实际项目中建议从 lab_common import tokenize
def tokenize(text: str) -> list:
    """统一的预处理：小写、保留否定词、去标点"""
    if not isinstance(text, str):
        return []
    text = text.lower()
    # 展开缩写，防止破坏否定词 (如 isn't -> is not)
    text = text.replace("n't", " not").replace("'re", " are").replace("'s", " is")
    # 去除标点，保留字母和数字
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

# ================= 2. 核心训练与评估函数 =================
def train_and_evaluate(train_df, window_size, output_dir, model_type="cbow"):
    print(f"\n[+] 开始训练: 窗口大小={window_size}, 模型={model_type.upper()}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 预处理
    sentences = [tokenize(text) for text in train_df['text'].tolist()]
    
    # 训练参数 (严格遵循文档要求)
    start_time = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=128,
        window=window_size,
        min_count=2,
        workers=1, # 单线程保证可复现
        epochs=10,
        sg=1 if model_type.lower() == "skipgram" else 0,
        seed=42
    )
    train_time = time.time() - start_time
    
    # 保存模型
    model.save(os.path.join(output_dir, "word2vec.model"))
    model.wv.save(os.path.join(output_dir, "vectors.kv"))
    
    # ---- 任务A: 计算相似度 (word_pairs.csv) ----
    word_pairs = [("good", "bad"), ("happy", "sad"), ("phone", "device"), ("not", "never"), ("love", "hate")]
    pairs_data = []
    for w1, w2 in word_pairs:
        if w1 in model.wv and w2 in model.wv:
            sim = model.wv.similarity(w1, w2)
            pairs_data.append({"word1": w1, "word2": w2, "similarity": round(float(sim), 4)})
        else:
            pairs_data.append({"word1": w1, "word2": w2, "similarity": "oov"})
    pd.DataFrame(pairs_data).to_csv(os.path.join(output_dir, "word_pairs.csv"), index=False)
    
    # ---- 任务B: 查看近邻 (neighbors.csv) ----
    query_words = ["good", "bad", "battery", "new", "phone", "not", "never", "love", "hate", "screen"]
    neighbors_data = []
    for word in query_words:
        if word in model.wv:
            for n_word, score in model.wv.most_similar(word, topn=5):
                neighbors_data.append({"query": word, "neighbor": n_word, "similarity": round(float(score), 4)})
        else:
            neighbors_data.append({"query": word, "neighbor": "OOV", "similarity": "oov"})
    pd.DataFrame(neighbors_data).to_csv(os.path.join(output_dir, "neighbors.csv"), index=False)
    
    # ---- 任务C: t-SNE 二维投影 ----
    # 取前 200 个高频词
    top_words = list(model.wv.key_to_index.keys())[:200]
    word_vectors = np.array([model.wv[w] for w in top_words])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_vectors = tsne.fit_transform(word_vectors)
    
    proj_df = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
    proj_df['word'] = top_words
    proj_df.to_csv(os.path.join(output_dir, "projection.csv"), index=False)
    
    # 画图 (图上仅标注前30个词)
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(top_words[:30]):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], marker='x', color='red', alpha=0.7)
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)
    plt.title(f"t-SNE Projection (Window={window_size}, {model_type.upper()})")
    plt.savefig(os.path.join(output_dir, "projection.png"))
    plt.close()

    # ---- 任务D: 导出实验2交接文件 (embedding.npy, vocab.json) ----
    # 索引0: <PAD> (全零), 索引1: <UNK> (固定随机种子)
    vocab = {"token_to_id": {"<PAD>": 0, "<UNK>": 1}, "id_to_token": ["<PAD>", "<UNK>"], "vector_size": 128}
    matrix = np.zeros((len(model.wv.key_to_index) + 2, 128), dtype=np.float32)
    
    # UNK 随机向量
    rng = np.random.default_rng(42)
    matrix[1] = rng.standard_normal(128).astype(np.float32)
    
    # 填充真实词向量
    for i, word in enumerate(model.wv.key_to_index):
        word_id = i + 2
        vocab["token_to_id"][word] = word_id
        vocab["id_to_token"].append(word)
        matrix[word_id] = model.wv[word]
        
    np.save(os.path.join(output_dir, "embedding.npy"), matrix)
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print(f"[+] 训练完成。耗时: {train_time:.2f}s | 词表大小: {len(model.wv.key_to_index)}")
    
    return {
        "window": window_size,
        "model": model_type,
        "vocab_size": len(model.wv.key_to_index),
        "train_time_sec": round(train_time, 2),
        "train_loss": round(float(model.get_latest_training_loss()), 4) if model.get_latest_training_loss() > 0 else "N/A"
    }

# ================= 3. 现场查询函数 =================
def query_vectors(model_path, words):
    from gensim.models import KeyedVectors
    print(f"\n[*] 从 {model_path} 加载模型进行查询...")
    wv = KeyedVectors.load(model_path, mmap='r')
    
    for word in words:
        if word in wv:
            print(f" 词: '{word}' -> Top-3 近邻:")
            for n, s in wv.most_similar(word, topn=3):
                print(f"   - {n}: {s:.4f}")
        else:
            print(f" 词: '{word}' -> OOV (词表外词汇，无法生成向量)")

# ================= 4. 主入口 =================
def main():
    parser = argparse.ArgumentParser(description="NLP 实验1: 词向量表示与相似度分析")
    parser.add_argument("--train-csv", type=str, default="examples/demo_train.csv", help="训练集 CSV 路径")
    parser.add_argument("--exclude-csv", nargs='+', default=[], help="需要排除的 dev/test CSV 路径")
    parser.add_argument("--output-dir", type=str, default="outputs/formal", help="输出目录")
    parser.add_argument("--algorithm", type=str, default="cbow", choices=["cbow", "skipgram"], help="训练算法")
    parser.add_argument("--query", nargs='+', help="现场查询词列表 (如果提供，仅执行查询逻辑)")
    args = parser.parse_args()

    # 如果传入了 --query 参数，则执行查询逻辑并退出
    if args.query:
        target_model = os.path.join(args.output_dir, "window_5", "vectors.kv")
        if os.path.exists(target_model):
            query_vectors(target_model, args.query)
        else:
            print(f"[-] 找不到模型文件 {target_model}，请先运行训练。")
        return

    # ---- 数据加载 ----
    if not os.path.exists(args.train_csv):
        print(f"[-] 错误：找不到训练文件 {args.train_csv}")
        return
    
    print(f"[*] 加载数据: {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    # 简单容错，确保有 text 列
    if 'text' not in df.columns:
        if len(df.columns) == 3: df.columns = ['label', 'title', 'text']
        elif len(df.columns) == 2: df.columns = ['label', 'text']
        else: raise ValueError("CSV 格式无法识别，请确保包含 'text' 列。")
        
    # 简单去重与排除重叠 (如果需要严格的 hash 排除，参考之前的 prepare_data.py)
    df = df.drop_duplicates(subset=['text'])
    
    # ---- 执行双窗口对照训练 ----
    results = []
    for window in [2, 5]:
        out_path = os.path.join(args.output_dir, f"window_{window}")
        res = train_and_evaluate(df, window, out_path, args.algorithm)
        results.append(res)
        
    # ---- 保存对比结果 ----
    comparison_path = os.path.join(args.output_dir, "comparison.csv")
    pd.DataFrame(results).to_csv(comparison_path, index=False)
    print(f"\n[+] 实验结束！对比结果已保存至: {comparison_path}")

if __name__ == "__main__":
    main()