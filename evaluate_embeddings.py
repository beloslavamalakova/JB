import json
import numpy as np


def load_vocab(word2id_path: str):
    with open(word2id_path, "r", encoding="utf-8") as f:
        word2id = json.load(f)
    id2word = {int(i): w for w, i in word2id.items()}
    return word2id, id2word


def cosine_sim_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    vec: (D,)
    mat: (V, D)
    returns: (V,) cosine similarities
    """
    vec_norm = np.linalg.norm(vec) + 1e-10
    mat_norm = np.linalg.norm(mat, axis=1) + 1e-10
    return (mat @ vec) / (mat_norm * vec_norm)


def nearest_neighbors(query_word: str, V: np.ndarray, word2id: dict, id2word: dict, topk: int = 10):
    if query_word not in word2id:
        raise ValueError(f"Word '{query_word}' not in vocabulary.")

    qid = word2id[query_word]
    qvec = V[qid]

    sims = cosine_sim_matrix(qvec, V)

    # exclude the query word itself
    sims[qid] = -np.inf

    nn_ids = np.argsort(-sims)[:topk]
    return [(id2word[int(i)], float(sims[i])) for i in nn_ids]


def main():
    V = np.load("embeddings_V.npy")  # input embeddings (center)
    word2id, id2word = load_vocab("word2id.json")

    # a few words expect to exist in the emotions dataset
    test_words = ["joy", "sadness"]

    for w in test_words:
        if w in word2id:
            print(f"\nnearest neighbors for '{w}'")
            for nn_word, sim in nearest_neighbors(w, V, word2id, id2word, topk=10):
                print(f"{nn_word:15s}  {sim:.4f}")
        else:
            print(f"\n[skip] '{w}' not in vocab")


if __name__ == "__main__":
    main()
