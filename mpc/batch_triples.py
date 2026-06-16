import json
from pathlib import Path

from triple import generate_triple, start_backend, stop_backend

N = 3
ell = 4

triples_dir = Path("triples")
triples_dir.mkdir(exist_ok=True)

alice_path = triples_dir / "alice.json"
bob_path = triples_dir / "bob.json"

alice_triples = json.load(open(alice_path)) if alice_path.exists() else []
bob_triples = json.load(open(bob_path)) if bob_path.exists() else []

start_backend()
try:
    for i in range(N):
        (u0, v0, z0), (u1, v1, z1) = generate_triple(ell)
        alice_triples.append({"u": int(u0), "v": int(v0), "z": int(z0), "ell": int(ell)})
        bob_triples.append({"u": int(u1), "v": int(v1), "z": int(z1), "ell": int(ell)})

        mod = 2 ** ell
        ok = ((u0 + u1) * (v0 + v1)) % mod == (z0 + z1) % mod
        print(f"[{i}] match={ok}")

        # save incrementally so we don't lose this run's progress
        json.dump(alice_triples, open(alice_path, "w"))
        json.dump(bob_triples, open(bob_path, "w"))
finally:
    stop_backend()

print(f"Total triples now: {len(alice_triples)}")