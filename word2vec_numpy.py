import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid -- scores become numbers in (0,1), like “probability this pair is real”
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))


# dataclass is a Python decorator that automatically creates a simple container class
@dataclass
class Word2VecConfig:
    dim: int = 100
    negative_k: int = 5
    lr: float = 0.025
    epochs: int = 3
    batch_size: int = 1
    seed: int = 42


class SkipGramNegSampling:
    """
    Skip-gram with Negative Sampling (SGNS) in pure NumPy.
    Two embedding matrices:
      V: input embeddings  (center words)
      U: output embeddings (context words)
    """

    def __init__(self, vocab_size: int, config: Word2VecConfig):
        self.vocab_size = vocab_size
        self.cfg = config
        rng = np.random.default_rng(config.seed)

        # Initialize embeddings (small uniform range)
        bound = 0.5 / config.dim
        self.V = rng.uniform(-bound, bound, size=(vocab_size, config.dim)).astype(np.float32)
        self.U = rng.uniform(-bound, bound, size=(vocab_size, config.dim)).astype(np.float32)

    def sample_negatives(self, neg_dist: np.ndarray, k: int, forbidden_id: int) -> np.ndarray:
        """
        Samples k negatives. We avoid sampling the positive context id by resampling if needed.
        """
        neg_ids = np.random.choice(self.vocab_size, size=k, p=neg_dist)
        # resample any forbidden ids
        while True:
            mask = (neg_ids == forbidden_id)
            if not mask.any():
                break
            neg_ids[mask] = np.random.choice(self.vocab_size, size=int(mask.sum()), p=neg_dist)
        return neg_ids.astype(np.int32)

    def train_one_pair(self, center_id: int, context_id: int, neg_dist: np.ndarray) -> float:
        """
        One SGD update for one (center, context) pair.
        Returns scalar loss.
        """
        k = self.cfg.negative_k

        v_c = self.V[center_id]          # (D,)
        u_o = self.U[context_id]         # (D,)

        neg_ids = self.sample_negatives(neg_dist, k, forbidden_id=context_id)
        u_negs = self.U[neg_ids]         # (k, D)

        # ---- forward ----
        s_pos = np.dot(u_o, v_c)         # scalar
        s_negs = u_negs @ v_c            # (k,)

        sig_pos = sigmoid(s_pos)         # scalar
        sig_negs = sigmoid(s_negs)       # (k,)

        # loss = -log sigma(s_pos) - sum log sigma(-s_neg)
        # log sigma(-x) = log(1-sigma(x))
        eps = 1e-10
        loss = -np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_negs + eps))

        # ---- backward ----
        # dL/ds_pos = sigma(s_pos) - 1
        grad_s_pos = (sig_pos - 1.0)     # scalar

        # dL/ds_neg = sigma(s_neg)  (for each neg)
        grad_s_negs = sig_negs           # (k,)

        # gradients
        # dL/dv_c = (sigma(s_pos)-1) u_o + sum sigma(s_neg_i) u_neg_i
        grad_v = grad_s_pos * u_o + (grad_s_negs[:, None] * u_negs).sum(axis=0)  # (D,)

        # dL/du_o = (sigma(s_pos)-1) v_c
        grad_uo = grad_s_pos * v_c       # (D,)

        # dL/du_neg_i = sigma(s_neg_i) v_c
        grad_unegs = grad_s_negs[:, None] * v_c[None, :]  # (k, D)

        # SGD update
        lr = self.cfg.lr

        # Important: update V[center_id] using grad_v
        self.V[center_id] -= lr * grad_v.astype(np.float32)

        # Update U for positive context
        self.U[context_id] -= lr * grad_uo.astype(np.float32)

        # Update U for negatives; handle duplicates safely
        # (np.add.at accumulates, but we need subtract -> add negative gradient)
        np.add.at(self.U, neg_ids, -lr * grad_unegs.astype(np.float32))

        return float(loss)

    def fit(self, pairs: List[Tuple[int, int]], neg_dist: np.ndarray, verbose_every: int = 200000) -> None:
        rng = np.random.default_rng(self.cfg.seed)

        for epoch in range(self.cfg.epochs):
            rng.shuffle(pairs)

            total_loss = 0.0
            for step, (c, o) in enumerate(pairs, start=1):
                total_loss += self.train_one_pair(c, o, neg_dist)

                if verbose_every and step % verbose_every == 0:
                    avg = total_loss / step
                    print(f"[epoch {epoch+1}/{self.cfg.epochs}] step {step:,} avg_loss={avg:.4f}")

            avg_epoch = total_loss / max(1, len(pairs))
            print(f"Epoch {epoch+1} finished. avg_loss={avg_epoch:.4f}")

    def get_input_embeddings(self) -> np.ndarray:
        return self.V

    def get_output_embeddings(self) -> np.ndarray:
        return self.U
