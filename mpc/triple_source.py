"""Pluggable Beaver-triple sources for the secure trainer.

Two backends, selectable at runtime so the training loop never has to care
where its triples come from:

  - ClassicalTripleSource : fresh pseudo-random triples generated on the fly.
  - QuantumFileTripleSource: pre-generated triples loaded from
    <dir>/{alice,bob}.json and consumed in order. This is the path that used to
    be commented out in train.py; it lets a run use the quantum-generated
    triples in triples/ (produced offline via SimulaQron) instead of classical
    ones.

Both yield a triple in the exact shape beaver_mul expects:

    ((a0, b0, c0), (a1, b1, c1))     with   (a0+a1)(b0+b1) == (c0+c1)  (mod 2**ell)
"""

import json
from collections import deque
from pathlib import Path


class TripleSource:
    """Common interface. `ell` is the bit-width the triples are valid for."""
    ell = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def next(self):
        raise NotImplementedError


class ClassicalTripleSource(TripleSource):
    """Pseudo-random Beaver triples, generated one per call (the default)."""

    def __init__(self, ell):
        from mpc.triples_generator import generate_classical_triple
        self.ell = ell
        self._gen = generate_classical_triple

    def next(self):
        return self._gen(self.ell)


class QuantumFileTripleSource(TripleSource):
    """Pre-generated triples from <dir>/{alice,bob}.json.

    The file dictates `ell` (the trainer adopts it). The file is finite, so by
    default we recycle once exhausted -- handy for a demo, but note that
    *reusing* Beaver triples breaks the privacy guarantee, hence the warning.
    """

    def __init__(self, triples_dir, ell=None, recycle=True):
        triples_dir = Path(triples_dir)
        alice = json.load(open(triples_dir / "alice.json"))
        bob = json.load(open(triples_dir / "bob.json"))
        if not alice or not bob:
            raise ValueError(f"No triples found in {triples_dir}")

        file_ell = alice[0]["ell"]
        if any(t["ell"] != file_ell for t in alice) or any(t["ell"] != file_ell for t in bob):
            raise ValueError(f"Mixed ell values in {triples_dir}; not supported")
        if ell is not None and ell != file_ell:
            print(f"[triple-source] requested ell={ell} but file triples are "
                  f"ell={file_ell}; using ell={file_ell}.")

        self.ell = file_ell
        self._alice = alice
        self._bob = bob
        self._aq = deque(alice)
        self._bq = deque(bob)
        self._recycle = recycle
        self._warned = False
        self._used = 0

    def next(self):
        if not self._aq:
            if not self._recycle:
                raise RuntimeError(
                    f"Exhausted {len(self._alice)} triples after {self._used} uses")
            if not self._warned:
                print(f"[triple-source] only {len(self._alice)} triples available; "
                      f"recycling them. Reusing Beaver triples is INSECURE (demo only).")
                self._warned = True
            self._aq = deque(self._alice)
            self._bq = deque(self._bob)
        a = self._aq.popleft()
        b = self._bq.popleft()
        self._used += 1
        return ((a["u"], a["v"], a["z"]), (b["u"], b["v"], b["z"]))


def make_triple_source(kind, ell, triples_dir="triples", recycle=True):
    if kind == "classical":
        return ClassicalTripleSource(ell)
    if kind == "quantum":
        return QuantumFileTripleSource(triples_dir, ell, recycle=recycle)
    raise ValueError(f"unknown triple source: {kind!r}")
