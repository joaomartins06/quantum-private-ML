import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from mpc.multiply import (
    encode, decode, share, reconstruct,
    load_triples, next_triple, beaver_mul, truncate
)

ELL = 64
F = 16
MOD = 1 << ELL
LR = 0.001
BATCH_SIZE = 4
EPOCHS = 3
TRIPLES_DIR = Path("triples_classical")
#Division of features between Alice and Bob (there are 10 features in this dataset)
ALICE_FEATURES = list(range(0, 5))   
BOB_FEATURES   = list(range(5, 10))


def prepare_data():
    data = load_diabetes()
    X = data.data
    y = data.target
    #normalized data
    y = (y - y.mean()) / y.std()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def encode_and_share(M, f, ell):
    mod = 1 << ell
    if M.ndim == 1:
        s0, s1 = [], []
        for x in M:
            enc = encode(float(x), f) % mod
            a, b = share(int(enc), ell)
            s0.append(int(a))
            s1.append(int(b))
        return s0, s1
    else:
        s0, s1 = [], []
        for row in M:
            r0, r1 = [], []
            for x in row:
                enc = encode(float(x), f) % mod
                a, b = share(int(enc), ell)
                r0.append(int(a))
                r1.append(int(b))
            s0.append(r0)
            s1.append(r1)
        return s0, s1


def decode_and_reconstruct(s0, s1, f, ell):
    flat_s0 = np.array(s0).flatten()
    flat_s1 = np.array(s1).flatten()
    M_reconstructed = np.array([reconstruct(x, y, ell) for x, y in zip(flat_s0, flat_s1)])
    M_decoded = np.vectorize(lambda x: decode(x, f, ell))(M_reconstructed)
    return M_decoded.reshape(np.array(s0).shape)


def mul_by_cte(x_sh, c_sh, f, ell):
    #multiply by a constant
    mod = 1 << ell
    x_sh_new  = ((x_sh[0] * c_sh) % mod, (x_sh[1] * c_sh) % mod)
    return truncate(x_sh_new, f, ell)


def secure_forward(X_sh0, X_sh1, w_sh, batch_idx, d, alice_q, bob_q):
    pred_sh = []
    for i in batch_idx:
        pred = (0, 0)
        for j in range(d):
            triple = next_triple(alice_q, bob_q)
            x_sh = (X_sh0[i][j], X_sh1[i][j])
            #this is where we actualliy make use of the MPC multiplication protocol
            prod_sh = beaver_mul(x_sh, w_sh[j], triple, ELL)
            prod_sh = truncate(prod_sh, F, ELL)
            pred = ((pred[0] + prod_sh[0]) % MOD, (pred[1] + prod_sh[1]) % MOD)
        pred_sh.append(pred)    
    return pred_sh


def secure_backward(X_sh0, X_sh1, r_sh, batch_idx, d, alice_q, bob_q):
    grad_sh = []
    for j in range(d):
        grad = (0, 0)
        for k, i in enumerate(batch_idx):
            triple = next_triple(alice_q, bob_q)
            x_sh = (X_sh0[i][j], X_sh1[i][j])
            prod_sh = beaver_mul(x_sh, r_sh[k], triple, ELL)
            prod_sh = truncate(prod_sh, F, ELL)
            grad = ((grad[0] + prod_sh[0]) % MOD, (grad[1] + prod_sh[1]) % MOD)
        grad_sh.append(mul_by_cte(grad, encode(1.0/len(batch_idx), F) % MOD, F, ELL))
    return grad_sh


def update_weights(w_sh, grad_sh, lr_enc, d):
    mod = 1 << ELL
    w_new = []
    for j in range(d):
        lr_g = mul_by_cte(grad_sh[j], lr_enc, F, ELL)
        w_new.append(((w_sh[j][0] - lr_g[0]) % mod, (w_sh[j][1] - lr_g[1]) % mod))
    return w_new


def loss_function(y_true, X, w):
    #mean squared error
    y_pred = X @ w
    return np.mean((y_true - y_pred) ** 2)


def train():
    X_train, X_test, y_train, y_test = prepare_data()
    n, d = X_train.shape

    X_alice = X_train[:, ALICE_FEATURES]
    X_bob   = X_train[:, BOB_FEATURES]

    X_sh0_alice, X_sh1_alice = encode_and_share(X_alice, F, ELL)
    X_sh0_bob, X_sh1_bob = encode_and_share(X_bob, F, ELL)
    y_sh0, y_sh1 = encode_and_share(y_train, F, ELL)

    X_sh0 = [X_sh0_alice[i] + X_sh0_bob[i] for i in range(n)]
    X_sh1 = [X_sh1_alice[i] + X_sh1_bob[i] for i in range(n)]

    #initialize weights
    w_sh = [(0, 0) for _ in range(d)]

    #load triples
    alice_q = load_triples(TRIPLES_DIR / "alice.json")
    bob_q = load_triples(TRIPLES_DIR / "bob.json")

    lr_enc = encode(LR, F) % MOD

    loss_history = []
    w_plain = [decode(reconstruct(w_sh[j][0], w_sh[j][1], ELL), F, ELL) for j in range(d)]
    loss = loss_function(y_train, X_train, np.array(w_plain))
    loss_history.append(loss)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Triples left: {len(alice_q)}")

        for batch_start in range(0, n, BATCH_SIZE):
            batch_idx = list(range(batch_start, min(batch_start + BATCH_SIZE, n)))
            pred_sh = secure_forward(X_sh0, X_sh1, w_sh, batch_idx, d, alice_q, bob_q)
            r_sh = []

            for k, i in enumerate(batch_idx):
                r0 = (pred_sh[k][0] - y_sh0[i]) % MOD
                r1 = (pred_sh[k][1] - y_sh1[i]) % MOD
                r_sh.append((r0, r1))

            grad_sh = secure_backward(X_sh0, X_sh1, r_sh, batch_idx, d, alice_q, bob_q)
            w_sh = update_weights(w_sh, grad_sh, lr_enc, d)

        w_plain = [decode(reconstruct(w_sh[j][0], w_sh[j][1], ELL), F, ELL) for j in range(d)]
        loss = loss_function(y_train, X_train, np.array(w_plain))
        loss_history.append(loss)
    
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Secure SGD Training Loss (MPC over Beaver Triples)")
    plt.grid(True)
    plt.savefig("figs/loss_history.png")
    plt.show()
    print("Loss plot saved to figs/loss_history.png")

    # both parties exchange shares and reconstruct w. the weights are not really private
    w_plain = [decode(reconstruct(w_sh[j][0], w_sh[j][1], ELL), F, ELL) for j in range(d)]
    print("\nFinal weights:")
    for j in range(d):
        owner = "Alice" if j < 5 else "Bob"
        print(f"  w[{j}] ({owner} feat {j if j < 5 else j-5}) = {w_plain[j]:.6f}")

    test_mse = loss_function(y_test, X_test, np.array(w_plain))
    print(f"\nTest MSE (secure): {test_mse:.6f}")

    # sklearn baseline
    sgd = SGDRegressor(max_iter=EPOCHS * (n // BATCH_SIZE), learning_rate='constant',
                       eta0=LR, random_state=42, fit_intercept=False, tol=None)
    sgd.fit(X_train, y_train)
    sklearn_mse = loss_function(y_test, X_test, sgd.coef_)
    print(f"Test MSE (sklearn baseline): {sklearn_mse:.6f}")

    
if __name__ == "__main__":
    train()