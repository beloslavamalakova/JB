# Task: 
Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the code, gradient derivation, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.

# Notes: 
Order of running is (data is in the folder ~/data):
1. data_preprocessing.py
2. word2vec_numpy.py
3. train.py
4. evaluate_embeddings.py

## data_preprocessing.py and word2vec_numpy.py
detailed comments in code

## train.py
| Epoch | Average Loss |
| ----- | ------------ |
| 1     | 3.2030       |
| 2     | 2.6500       |
| 3     | 2.5679       |


## evaluate_embeddings.py
### Nearest neighbors for `joy`

| Rank | Word      | Cosine Similarity |
| ---- | --------- | ----------------- |
| 1    | worthy    | 0.9876            |
| 2    | repeat    | 0.9866            |
| 3    | satan     | 0.9859            |
| 4    | clothes   | 0.9851            |
| 5    | despised  | 0.9848            |
| 6    | slutty    | 0.9843            |
| 7    | tragic    | 0.9836            |
| 8    | supported | 0.9835            |
| 9    | vile      | 0.9835            |
| 10   | talented  | 0.9822            |

### Nearest neighbors for `sadness`

| Rank | Word      | Cosine Similarity |
| ---- | --------- | ----------------- |
| 1    | poetry    | 0.9918            |
| 2    | touch     | 0.9918            |
| 3    | focused   | 0.9914            |
| 4    | although  | 0.9906            |
| 5    | broken    | 0.9889            |
| 6    | missing   | 0.9885            |
| 7    | badly     | 0.9879            |
| 8    | concerned | 0.9878            |
| 9    | core      | 0.9875            |
| 10   | balance   | 0.9870            |

