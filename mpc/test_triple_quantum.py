"""
End-to-end test for quantum Beaver-triple generation (Layer 2).

This drives the real ``triple.py`` -> ``alice_ot.py`` / ``bob_ot.py`` pipeline
over a live SimulaQron backend and checks the defining Beaver relation:

    (u0 + u1) * (v0 + v1)  ==  z0 + z1   (mod 2^ell)

It is SLOW (every OT simulates qubits) and needs the quantum stack, so run it
from the project's ``simulaqron-venv``:

    pytest mpc/test_triple_quantum.py -v

The module is skipped automatically if the ``simulaqron`` CLI is not on PATH,
so it stays green in environments without the quantum backend.
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc.triple import start_backend, stop_backend, generate_triple

ELL = 4  # keep small: SimulaQron simulates ~2*ELL OTs of ELL qubits per triple

pytestmark = pytest.mark.skipif(
    shutil.which("simulaqron") is None,
    reason="requires the SimulaQron backend; run inside the quantum env",
)


@pytest.fixture(scope="module")
def backend():
    """Start one SimulaQron backend for the whole module, stop it after."""
    start_backend()
    try:
        yield
    finally:
        stop_backend()


def test_quantum_triple_satisfies_beaver_relation(backend):
    mod = 1 << ELL
    # a few independent triples: the relation must hold for every one
    for _ in range(3):
        (u0, v0, z0), (u1, v1, z1) = generate_triple(ELL)
        lhs = ((u0 + u1) % mod) * ((v0 + v1) % mod) % mod
        rhs = (z0 + z1) % mod
        assert lhs == rhs, f"Beaver relation failed: {lhs} != {rhs}"
