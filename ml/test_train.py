"""
Convergence test for the secure linear regression pipeline (Layer 3).

Runs a few epochs of secure mini-batch SGD on a small synthetic dataset using
the real ``train.py`` primitives (encode/share, secure forward/backward, weight
update) and checks two things:

  1. the secure model converges (final loss << initial loss), and
  2. its weights match a plaintext SGD doing the identical update math, to
     within fixed-point + truncation error.

This validates that the forward/backward/update steps compose correctly. It is
classical (uses ``generate_classical_triple``, no quantum backend), so it runs
in well under a second:

    pytest ml/test_train.py -v
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.train import (
    encode, decode, reconstruct, encode_and_share,
    secure_forward, secure_backward, update_weights,
)
from mpc.triple_source import make_triple_source

ELL = 64
F = 16
MOD = 1 << ELL
LR = 0.05
BATCH_SIZE = 4
EPOCHS = 30


@pytest.fixture(autouse=True)
def _seed():
    # fix seeds so share randomness and triple generation are deterministic
    np.random.seed(0)
    random.seed(0)


def make_data():
    # small, well-conditioned linear problem with no noise so convergence is fast
    n, d = 12, 3
    X = np.random.randn(n, d)
    w_true = np.array([1.5, -2.0, 0.5])
    y = X @ w_true
    return X, y, w_true


def mse(X, y, w):
    return float(np.mean((y - X @ w) ** 2))


def reconstruct_weights(w_sh, d):
    # recover plaintext weights from shares for comparison
    return np.array([
        decode(reconstruct(w_sh[j][0], w_sh[j][1], ELL), F, ELL) for j in range(d)
    ])


def run_secure_sgd(X, y):
    # run the real train.py primitives end-to-end with on-demand classical triples
    n, d = X.shape
    X_sh0, X_sh1 = encode_and_share(X, F, ELL)
    y_sh0, y_sh1 = encode_and_share(y, F, ELL)
    w_sh = [(0, 0) for _ in range(d)]
    lr_enc = encode(LR, F) % MOD
    triples = make_triple_source("classical", ELL)

    for _ in range(EPOCHS):
        for start in range(0, n, BATCH_SIZE):
            batch_idx = list(range(start, min(start + BATCH_SIZE, n)))
            pred_sh = secure_forward(X_sh0, X_sh1, w_sh, batch_idx, d, triples, F, ELL)
            # residual computed locally from shares, no triples needed
            r_sh = [
                ((pred_sh[k][0] - y_sh0[i]) % MOD, (pred_sh[k][1] - y_sh1[i]) % MOD)
                for k, i in enumerate(batch_idx)
            ]
            grad_sh = secure_backward(X_sh0, X_sh1, r_sh, batch_idx, d, triples, F, ELL)
            w_sh = update_weights(w_sh, grad_sh, lr_enc, d, F, ELL)

    return reconstruct_weights(w_sh, d)


def run_plaintext_sgd(X, y):
    # identical update math in plaintext, used as ground truth to verify the secure version
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(EPOCHS):
        for start in range(0, n, BATCH_SIZE):
            idx = slice(start, min(start + BATCH_SIZE, n))
            Xb, yb = X[idx], y[idx]
            r = Xb @ w - yb
            grad = (Xb.T @ r) / Xb.shape[0]
            w = w - LR * grad
    return w


def test_secure_training_converges():
    X, y, _ = make_data()
    w_secure = run_secure_sgd(X, y)

    # loss should drop to at least 10% of the value at w=0
    initial_loss = mse(X, y, np.zeros(X.shape[1]))
    final_loss = mse(X, y, w_secure)
    assert final_loss < initial_loss * 0.1, (
        f"secure training did not converge: {initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_secure_matches_plaintext_sgd():
    X, y, _ = make_data()
    w_secure = run_secure_sgd(X, y)
    w_plain = run_plaintext_sgd(X, y)
    # only fixed-point quantisation and ±1-LSB truncation error should separate them
    assert np.allclose(w_secure, w_plain, atol=1e-2), (
        f"secure {w_secure} vs plaintext {w_plain}"
    )