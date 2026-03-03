import numpy as np
import re  # regular expressions for cleaning text
from collections import Counter
from typing import List, Tuple, Dict
import nltk  # used to download and use the stopwords
from nltk.corpus import stopwords

# the dataset used is: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
# the dataset is in folder ~/data, and is ignored in the .gitignore file in order not to post data on github, following good practices

# I used the NLTK stopword list to ensure a standardized and reproducible rather than manually defining
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


# I normalize text by making everything lowercase
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove anything that is NOT a-z or whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    # split on spaces into a list of tokens
    return text.split()


def load_dataset(path: str) -> List[List[str]]:
    """
    Dataset format: sentence;text_label
    """
    sentences = []  # this will store tokenized sentences
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # If the dataset contains a label separated by ";", split it off
            if ";" in line:
                text, _ = line.rsplit(";", 1)
            else:
                text = line

            text = normalize_text(text)
            tokens = tokenize(text)

            # remove stopwords
            tokens = [w for w in tokens if w not in STOPWORDS]

            if tokens:
                sentences.append(tokens)

    return sentences


def build_vocab(sentences: List[List[str]], min_count: int = 2):
    counter = Counter() # to store word counts
    for sentence in sentences:
        counter.update(sentence)

    vocab = [word for word, count in counter.items() if count >= min_count]  # filtering out rare words
    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for word, idx in word2id.items()}
    print(f"vocabulary size: {len(vocab)}")

    return word2id, id2word, counter  # return mappings and counts


def sentences_to_ids(sentences, word2id):
    corpus = []
    for sentence in sentences:
        ids = [word2id[word] for word in sentence if word in word2id]
        corpus.extend(ids)

    return np.array(corpus, dtype=np.int32)


def generate_skipgram_pairs(corpus_ids, window_size=2):
    pairs = []

    for i in range(len(corpus_ids)):
        center = corpus_ids[i]

        for j in range(-window_size, window_size + 1):
            if j == 0:
                continue

            context_index = i + j
            if 0 <= context_index < len(corpus_ids):
                context = corpus_ids[context_index]
                pairs.append((center, context))

    return pairs


def build_negative_sampling_dist(counter, word2id, power=0.75):
    vocab_size = len(word2id)
    freqs = np.zeros(vocab_size)

    for word, idx in word2id.items():
        freqs[idx] = counter[word]

    freqs = freqs ** power
    dist = freqs / np.sum(freqs)

    return dist


def prepare_data(train_path: str, min_count: int = 2, window_size: int = 2):
    sentences = load_dataset(train_path)  # read file -> tokenized sentences (stopwords removed)
    word2id, id2word, counter = build_vocab(sentences, min_count=min_count)  # build vocab and word counts
    corpus_ids = sentences_to_ids(sentences, word2id)  # convert tokens to a flat stream of ids
    pairs = generate_skipgram_pairs(corpus_ids, window_size=window_size)  # build training pairs
    neg_dist = build_negative_sampling_dist(counter, word2id)  # build sampling distribution for negatives
    print(f"Total training pairs: {len(pairs)}")
    return pairs, word2id, id2word, neg_dist
