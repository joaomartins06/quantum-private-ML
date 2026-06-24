import sys
import argparse
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from mpc.multiply import (
    encode, decode, share, reconstruct, beaver_mul, truncate
)
from mpc.triple_source import make_triple_source

# division of features between alice and bob (10 features total in the diabetes dataset)
# alice holds demographics and physical measurements (features 0-4)
# bob holds blood serum measurements (features 5-9)
ALICE_FEATURES = list(range(0, 5))
BOB_FEATURES   = list(range(5, 10))


def prepare_data(max_samples=None):
    data = load_diabetes()
    X = data.data
    y = data.target
    # normalize y to zero mean, unit variance so gradients stay in a reasonable range
    y = (y - y.mean()) / y.std()
    # standardize X so all features are on the same scale before encoding
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    if max_samples is not None:
        # handy to shrink the problem for the slow quantum-live source
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
    return X_train, X_test, y_train, y_test


def encode_and_share(M, f, ell):
    # convert every real value to a fixed-point integer scaled by 2^f,
    # then split into two additive shares s0, s1 such that s0 + s1 = enc mod 2^ell
    # returns python lists (not numpy arrays) to avoid int64 overflow at ell=64
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
        # matrix case: return a list of rows, one share-row per party
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
    # utility to recover plaintext values from shares, used for monitoring only
    flat_s0 = np.array(s0).flatten()
    flat_s1 = np.array(s1).flatten()
    M_reconstructed = np.array([reconstruct(x, y, ell) for x, y in zip(flat_s0, flat_s1)])
    M_decoded = np.vectorize(lambda x: decode(x, f, ell))(M_reconstructed)
    return M_decoded.reshape(np.array(s0).shape)


def mul_by_cte(x_sh, c_sh, f, ell):
    # multiply a shared value by a public constant, no triple needed
    # each party scales its own share locally, then truncate removes the extra 2^f factor
    mod = 1 << ell
    x_sh_new  = ((x_sh[0] * c_sh) % mod, (x_sh[1] * c_sh) % mod)
    return truncate(x_sh_new, f, ell)


def secure_forward(X_sh0, X_sh1, w_sh, batch_idx, d, triples, f, ell):
    # compute pred_i = X[i] . w for each sample in the batch, entirely in shares
    # outer loop: one prediction per sample
    # inner loop: dot product over d features, each multiply consumes one beaver triple
    mod = 1 << ell
    pred_sh = []
    for i in batch_idx:
        # accumulate the dot product for sample i as a share pair
        pred = (0, 0)
        for j in range(d):
            triple = triples.next()
            # pack the two shares of X[i][j] into the format beaver_mul expects
            x_sh = (X_sh0[i][j], X_sh1[i][j])
            # secure multiply: core MPC primitive, privacy guaranteed by the triple
            prod_sh = beaver_mul(x_sh, w_sh[j], triple, ell)
            # remove the extra 2^f scaling introduced by multiplying two fixed-point values
            prod_sh = truncate(prod_sh, f, ell)
            # add this feature's contribution to the running dot product
            pred = ((pred[0] + prod_sh[0]) % mod, (pred[1] + prod_sh[1]) % mod)
        pred_sh.append(pred)
    return pred_sh


def secure_backward(X_sh0, X_sh1, r_sh, batch_idx, d, triples, f, ell):
    # compute grad_j = (1/B) * sum_i X[i][j] * r[i] for each feature j, all in shares
    # r_sh is the residual (pred - y), already in shares from the training loop
    # outer loop: one gradient component per feature
    # inner loop: accumulate over all batch samples, each multiply consumes one beaver triple
    mod = 1 << ell
    grad_sh = []
    for j in range(d):
        # accumulate gradient for feature j across all batch samples
        grad = (0, 0)
        for k, i in enumerate(batch_idx):
            triple = triples.next()
            x_sh = (X_sh0[i][j], X_sh1[i][j])
            # r_sh is indexed by k (position in batch), not i (row index in X_train)
            prod_sh = beaver_mul(x_sh, r_sh[k], triple, ell)
            prod_sh = truncate(prod_sh, f, ell)
            grad = ((grad[0] + prod_sh[0]) % mod, (grad[1] + prod_sh[1]) % mod)
        # divide by batch size using a public constant multiply, no triple needed
        grad_sh.append(mul_by_cte(grad, encode(1.0 / len(batch_idx), f) % mod, f, ell))
    return grad_sh


def update_weights(w_sh, grad_sh, lr_enc, d, f, ell):
    # w = w - lr * grad, all local arithmetic on shares, no triples needed
    # lr_enc is the learning rate pre-encoded as a fixed-point integer
    mod = 1 << ell
    w_new = []
    for j in range(d):
        # scale gradient by lr using a public constant multiply
        lr_g = mul_by_cte(grad_sh[j], lr_enc, f, ell)
        # subtract from current weight shares
        w_new.append(((w_sh[j][0] - lr_g[0]) % mod, (w_sh[j][1] - lr_g[1]) % mod))
    return w_new


def loss_function(y_true, X, w):
    # standard MSE, used only for monitoring outside the secure protocol
    # neither party would compute this during a real deployment
    y_pred = X @ w
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    # coefficient of determination: 1.0 is perfect, 0.0 means no better than predicting the mean
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def train(args):
    X_train, X_test, y_train, y_test = prepare_data(args.max_samples)
    n, d = X_train.shape

    # build the triple source first: for quantum-file source it dictates ell, so we adopt it
    triples = make_triple_source(args.triples, args.ell, args.triples_dir)
    ell = triples.ell
    f = args.f
    mod = 1 << ell

    # sanity check: if f is too close to ell, products overflow the ring
    if f >= ell - 1:
        print(f"[train] WARNING: f={f} fixed-point bits with only ell={ell} bits leaves "
              f"no integer headroom; expect overflow. This config is a wiring demo, "
              f"not a meaningful fit.")

    # estimate triple budget upfront: forward pass uses B*d per batch, backward also B*d
    # total = 2 * n_batches * B * d * epochs = 2 * n * d * epochs
    est_triples = 2 * n * d * args.epochs
    print(f"[train] source={args.triples}  ell={ell}  f={f}  n={n}  d={d}  "
          f"epochs={args.epochs}  batch={args.batch_size}")
    print(f"[train] this run will consume ~{est_triples:,} Beaver triples.")

    # vertical split: alice encodes her columns, bob encodes his
    # neither party sees the other's raw data at any point
    X_alice = X_train[:, ALICE_FEATURES]
    X_bob   = X_train[:, BOB_FEATURES]

    # encode and share each party's feature matrix into additive shares
    X_sh0_alice, X_sh1_alice = encode_and_share(X_alice, f, ell)
    X_sh0_bob, X_sh1_bob = encode_and_share(X_bob, f, ell)
    # encode and share the labels (held by one party, e.g. alice, in a real deployment)
    y_sh0, y_sh1 = encode_and_share(y_train, f, ell)

    # concatenate alice and bob's feature shares into one full n x d share matrix
    # simulation convenience only: in reality each party only holds their own columns
    X_sh0 = [X_sh0_alice[i] + X_sh0_bob[i] for i in range(n)]
    X_sh1 = [X_sh1_alice[i] + X_sh1_bob[i] for i in range(n)]

    # initialize all weight shares to zero
    w_sh = [(0, 0) for _ in range(d)]
    # encode the learning rate once as a fixed-point integer
    lr_enc = encode(args.lr, f) % mod

    # record initial loss before any training (at w=0)
    loss_history = []
    w_plain = [decode(reconstruct(w_sh[j][0], w_sh[j][1], ell), f, ell) for j in range(d)]
    loss = loss_function(y_train, X_train, np.array(w_plain))
    loss_history.append(loss)

    # context manager starts the SimulaQron backend if using quantum-live triple source
    with triples:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

            for batch_start in range(0, n, args.batch_size):
                batch_idx = list(range(batch_start, min(batch_start + args.batch_size, n)))

                # forward pass: compute [pred_i] = [X[i]] . [w], consumes B*d triples
                pred_sh = secure_forward(X_sh0, X_sh1, w_sh, batch_idx, d, triples, f, ell)

                # residual [r] = [pred] - [y]: subtraction of shares is local, no triples
                r_sh = []
                for k, i in enumerate(batch_idx):
                    r0 = (pred_sh[k][0] - y_sh0[i]) % mod
                    r1 = (pred_sh[k][1] - y_sh1[i]) % mod
                    r_sh.append((r0, r1))

                # backward pass: compute [grad_j] = (1/B) sum_i [X[i][j]] * [r[i]], consumes B*d triples
                grad_sh = secure_backward(X_sh0, X_sh1, r_sh, batch_idx, d, triples, f, ell)

                # weight update: [w] = [w] - lr * [grad], local arithmetic only
                w_sh = update_weights(w_sh, grad_sh, lr_enc, d, f, ell)

            # reconstruct weights to monitor loss, this reveal would not happen in a real deployment
            w_plain = [decode(reconstruct(w_sh[j][0], w_sh[j][1], ell), f, ell) for j in range(d)]
            loss = loss_function(y_train, X_train, np.array(w_plain))
            loss_history.append(loss)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(0, args.epochs + 1), loss_history, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Secure SGD Training Loss ({args.triples} triples, ell={ell})")
        plt.grid(True)
        Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot_path)
        print(f"Loss plot saved to {args.plot_path}")

    # both parties exchange their weight shares and reconstruct w
    # this is the only point in the entire protocol where a non-masked value crosses party boundaries
    w_plain = [decode(reconstruct(w_sh[j][0], w_sh[j][1], ell), f, ell) for j in range(d)]
    print("\nFinal weights:")
    for j in range(d):
        owner = "Alice" if j < 5 else "Bob"
        print(f"  w[{j}] ({owner} feat {j if j < 5 else j-5}) = {w_plain[j]:.6f}")

    # evaluate the reconstructed model on held-out test data
    test_mse = loss_function(y_test, X_test, np.array(w_plain))
    print(f"\nTest MSE (secure): {test_mse:.6f}")

    # plaintext sklearn SGD with identical hyperparameters as a sanity baseline
    sgd = SGDRegressor(max_iter=args.epochs * max(1, n // args.batch_size),
                       learning_rate='constant', eta0=args.lr, random_state=42,
                       fit_intercept=False, tol=None)
    sgd.fit(X_train, y_train)
    sklearn_mse = loss_function(y_test, X_test, sgd.coef_)
    print(f"Test MSE (sklearn baseline): {sklearn_mse:.6f}")

    w_arr = np.array(w_plain)
    r2 = r2_score(y_test, X_test @ w_arr)
    print(f"R² (secure): {r2:.4f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Secure SGD linear regression over Beaver triples (classical or quantum).")
    p.add_argument("--triples", choices=["classical", "quantum"],
                   default="classical",
                   help="triple source: on-the-fly classical generation, or "
                        "pre-generated triples loaded from --triples-dir "
                        "(default: classical)")
    p.add_argument("--ell", type=int, default=64,
                   help="ring bit-width (ignored for --triples quantum, which uses the file's ell)")
    p.add_argument("--f", type=int, default=16, help="fixed-point fractional bits")
    p.add_argument("--lr", type=float, default=0.001, help="learning rate")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap the number of training rows")
    p.add_argument("--triples-dir", default="triples",
                   help="directory of pre-generated triples for --triples quantum")
    p.add_argument("--plot-path", default="figs/loss_history.png")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())