import json
from pathlib import Path

from triple import generate_triple, start_backend, stop_backend

N = 3
ell = 4

triples_dir = Path("triples")
triples_dir.mkdir(exist_ok=True)

alice_triples = []
bob_triples = []

start_backend()
try:
    for i in range(N):
        (u0, v0, z0), (u1, v1, z1) = generate_triple(ell)
        alice_triples.append({"u": int(u0), "v": int(v0), "z": int(z0), "ell": int(ell)})
        bob_triples.append({"u": int(u1), "v": int(v1), "z": int(z1), "ell": int(ell)})

        mod = 2 ** ell
        ok = ((u0 + u1) * (v0 + v1)) % mod == (z0 + z1) % mod
        print(f"[{i}] match={ok}")
finally:
    stop_backend()

with open(triples_dir / "alice.json", "w") as f:
    json.dump(alice_triples, f)
with open(triples_dir / "bob.json", "w") as f:
    json.dump(bob_triples, f)

print(f"Saved {len(alice_triples)} triples to {triples_dir}/alice.json and bob.json")