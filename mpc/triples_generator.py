import json
import random
from pathlib import Path


# simulaqron works, but it is extremely slow
# so we just generate triples locally with pure arithmetic, no privacy claim here


def share(x, ell):
    mod = 1 << ell
    # pick s0 at random, compute s1 so that s0 + s1 = x mod 2^ell
    s0 = random.randrange(mod)
    s1 = (x - s0) % mod
    return s0, s1


def generate_classical_triple(ell):
    mod = 1 << ell
    # pick random a, b and compute c = a*b directly
    a = random.randrange(mod)
    b = random.randrange(mod)
    c = (a * b) % mod

    # split each value into additive shares
    a0, a1 = share(a, ell)
    b0, b1 = share(b, ell)
    c0, c1 = share(c, ell)

    return (a0, b0, c0), (a1, b1, c1)


def generate_batch(N, ell, triples_dir):
    triples_dir = Path(triples_dir)
    triples_dir.mkdir(exist_ok=True)

    alice_path = triples_dir / "alice.json"
    bob_path = triples_dir / "bob.json"

    # load existing triples so we can accumulate across runs
    alice_triples = json.load(open(alice_path)) if alice_path.exists() else []
    bob_triples = json.load(open(bob_path)) if bob_path.exists() else []

    for i in range(N):
        (u0, v0, z0), (u1, v1, z1) = generate_classical_triple(ell)
        alice_triples.append({"u": int(u0), "v": int(v0), "z": int(z0), "ell": int(ell)})
        bob_triples.append({"u": int(u1), "v": int(v1), "z": int(z1), "ell": int(ell)})

        mod = 1 << ell
        ok = ((u0 + u1) * (v0 + v1)) % mod == (z0 + z1) % mod
        print(f"[{i}] match={ok}")

        # save after every triple so a crash only loses the one in flight
        json.dump(alice_triples, open(alice_path, "w"))
        json.dump(bob_triples, open(bob_path, "w"))

    print(f"Total triples now: {len(alice_triples)}")
    return alice_triples, bob_triples


if __name__ == "__main__":
    N = 10
    ELL = 64
    TRIPLES_DIR = "triples_classical"

    generate_batch(N, ELL, TRIPLES_DIR)