import numpy as np
import json
from data_preprocessing import prepare_data
from word2vec_numpy import SkipGramNegSampling, Word2VecConfig

def main():
    pairs, word2id, id2word, neg_dist = prepare_data(
        train_path="data/train.txt",
        min_count=2,
        window_size=2
    )

    cfg = Word2VecConfig(dim=100, negative_k=5, lr=0.025, epochs=3, seed=42)
    model = SkipGramNegSampling(vocab_size=len(word2id), config=cfg)

    model.fit(pairs, neg_dist, verbose_every=100000)

    V = model.get_input_embeddings()
    U = model.get_output_embeddings()

    np.save("embeddings_V.npy", V)
    np.save("embeddings_U.npy", U)

    # save vocab mapping for later use
    with open("word2id.json", "w", encoding="utf-8") as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)

    print("saved embeddings_V.npy, embeddings_U.npy, word2id.json")


if __name__ == "__main__":
    main()
