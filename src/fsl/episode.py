"""N-way K-shot Q-query episode sampler (reused for quantitative meta-learning).

Two split modes
---------------
sample-level split:
    All classes share a single sample pool. Each episode randomly draws n_way
    classes and K+Q distinct samples per class; the first K form the support
    set and the last Q form the query set. This is the primary protocol when
    the number of classes is small (<10).
class-level split (held-out class evaluation):
    The class set is partitioned into meta-train and meta-test classes;
    episodes are sampled only within the designated side, so meta-test classes
    are completely invisible during meta-training (leakage-free evaluation).

Reproducibility: the randomness of each episode is uniquely determined by
(seed, split, shot, rep); given the same parameters, any process reproduces
identical samples (enabling paired comparison across models).

Reuse convention: sample() returns global sample indices
(support_idx, query_idx); the caller slices X[idx] / y[idx] itself. For
regression, simply replace class labels with "formulation/interval buckets".
"""
from __future__ import annotations

import numpy as np


def class_counts(y):
    """Return a {class: sample count} dictionary."""
    vals, cnts = np.unique(np.asarray(y), return_counts=True)
    return dict(zip(vals.tolist(), cnts.tolist()))


def filter_rare_classes(X, y, min_count):
    """Drop classes with fewer than min_count samples.

    Returns (X_kept, y_kept, kept_classes, dropped_classes).
    Ensures every class can support K-shot + Q-query (min_count >= K + Q).
    """
    X, y = np.asarray(X), np.asarray(y)
    cnt = class_counts(y)
    kept = [c for c in cnt if cnt[c] >= min_count]
    dropped = [c for c in cnt if cnt[c] < min_count]
    mask = np.isin(y, kept)
    return X[mask], y[mask], kept, dropped


class EpisodeSampler:
    """N-way K-shot Q-query episode sampler.

    Parameters
    ----------
    y : (n_samples,) class labels (any hashable type)
    n_way : number of classes per episode; clipped to the number of available
        classes if it exceeds them
    k_shot : support-set size per class (sample() can temporarily override it
        via `shot`, for the 5/10/20-shot regimes)
    q_query : query-set size per class
    mode : "sample" (sample-level) | "class" (class-level held-out)
    train_classes / test_classes : class-level split for class mode; ignored
        in sample mode
    seed : base seed
    """

    def __init__(self, y, n_way, k_shot=5, q_query=5, mode="sample",
                 train_classes=None, test_classes=None, seed=42):
        y = np.asarray(y)
        self.y = y
        self.by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_query = int(q_query)
        self.mode = mode
        self.seed = int(seed)
        if mode == "class":
            if not train_classes or not test_classes:
                raise ValueError("class mode requires explicit train_classes and test_classes")
            self.train_classes = list(train_classes)
            self.test_classes = list(test_classes)
            unknown = (set(self.train_classes) | set(self.test_classes)) - set(self.by_class)
            if unknown:
                raise ValueError(f"class-level split contains unknown classes: {unknown}")
        elif mode == "sample":
            self.train_classes = self.test_classes = sorted(self.by_class, key=str)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        self._check(self.k_shot)

    # ---------- internals ----------
    def _check(self, K):
        for split in ("train", "test"):
            classes = self.classes_for(split)
            if len(classes) < 2:
                raise ValueError(f"{split} side has fewer than 2 classes; cannot build an episode")
            for c in classes:
                n = len(self.by_class[c])
                if n < K + self.q_query:
                    raise ValueError(
                        f"class {c!r} has only {n} samples, insufficient for K={K} + Q={self.q_query}")

    def _rng(self, split, shot, rep):
        s = (self.seed * 100003 + int(shot) * 1009 + int(rep) * 37
             + (0 if split == "train" else 7))
        return np.random.default_rng(s)

    # ---------- interface ----------
    def classes_for(self, split):
        """split: "train" | "test"; returns the list of classes available on that side."""
        return self.train_classes if split == "train" else self.test_classes

    def n_way_for(self, split):
        return min(self.n_way, len(self.classes_for(split)))

    def sample(self, split="test", rep=0, shot=None):
        """Sample one episode.

        Returns (support_idx, query_idx, episode_classes):
        - support_idx / query_idx: global sample indices, laid out contiguously
          per class (K / Q consecutive indices per class);
        - episode_classes: the class list of this episode, aligned one-to-one
          with the index blocks.
        """
        K = int(shot) if shot is not None else self.k_shot
        self._check(K)
        rng = self._rng(split, K, rep)
        pool = self.classes_for(split)
        n_way = min(self.n_way, len(pool))
        cls_pos = rng.choice(len(pool), size=n_way, replace=False)
        ep_classes = [pool[i] for i in cls_pos]
        sup, qry = [], []
        for c in ep_classes:
            idx = self.by_class[c]
            pick = rng.choice(len(idx), size=K + self.q_query, replace=False)
            sup.extend(idx[pick[:K]])
            qry.extend(idx[pick[K:]])
        return np.asarray(sup), np.asarray(qry), ep_classes

    def iter_episodes(self, n, split="train", shot=None, start_rep=0):
        """Generate n consecutive episodes (rep = start_rep .. start_rep+n-1)."""
        for r in range(start_rep, start_rep + n):
            yield self.sample(split=split, rep=r, shot=shot)
