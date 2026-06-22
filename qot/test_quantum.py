"""
End-to-end tests for the BB84 Oblivious Transfer (Layer 1).

These drive the real ``alice_ot.py`` / ``bob_ot.py`` over a live SimulaQron
backend and check that Bob recovers ``s_y`` (and only ``s_y``). They are SLOW
(every qubit is simulated) and require the quantum stack, so run them from the
project's ``simulaqron-venv``:

    pytest qot/test_quantum.py -v

The whole module is skipped automatically if the ``simulaqron`` CLI is not on
PATH, so it stays green in environments without the quantum backend.
"""

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

QOT = Path(__file__).parent
REPO = QOT.parent
ELL = 4  # keep small: SimulaQron simulates 4*ELL qubits per OT

# Skip everything here unless the SimulaQron backend is installed.
pytestmark = pytest.mark.skipif(
    shutil.which("simulaqron") is None,
    reason="requires the SimulaQron backend; run inside the quantum env",
)


def _bits_to_int(bit_string):
    """Little-endian, matching bob_ot.py: bit i carries weight 2**i."""
    bits = np.array([int(c) for c in bit_string], dtype=int)
    return int(bits @ (1 << np.arange(len(bits))))


def _expected(pairs, choices):
    return [
        _bits_to_int(s0 if y == 0 else s1)
        for (s0, s1), y in zip(pairs, choices)
    ]


@pytest.fixture(scope="module")
def backend():
    """Start one SimulaQron backend for the whole module, stop it after."""
    subprocess.run(["simulaqron", "stop"], capture_output=True)
    time.sleep(3.0)
    proc = subprocess.Popen(
        ["simulaqron", "start", "--nodes", "Alice,Bob",
         "--network-config-file", "qot/simulaqron_network.json"],
        cwd=REPO,
    )
    time.sleep(5.0)
    try:
        yield
    finally:
        subprocess.run(["simulaqron", "stop"], capture_output=True)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_ot_batch(pairs, choices, ell):
    """Run a batch of OTs: Bob (server) recovers s_y for each choice bit; Alice
    (client) supplies the (s0, s1) pairs. Returns Bob's recovered integers."""
    log_path = QOT / "bob_test.log"
    bob_log = open(log_path, "w+")
    bob_proc = subprocess.Popen(
        ["python", "bob_ot.py", str(ell), ",".join(str(y) for y in choices)],
        cwd=QOT, stdout=bob_log, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(2.0)  # let Bob start serving before Alice connects

    s_arg = ";".join(f"{s0},{s1}" for s0, s1 in pairs)
    alice = subprocess.run(
        ["python", "alice_ot.py", str(ell), s_arg],
        cwd=QOT, capture_output=True, text=True, timeout=600,
    )

    if alice.returncode != 0:
        bob_proc.terminate()
        bob_proc.wait()
        bob_log.close()
        raise AssertionError(f"Alice failed:\n{alice.stderr}")

    time.sleep(1.0)  # let Bob flush its RESULTS line
    bob_proc.terminate()
    try:
        bob_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        bob_proc.kill()
        bob_proc.wait()

    bob_log.seek(0)
    results = None
    for line in bob_log:
        if line.strip().startswith("RESULTS:"):
            results = [int(x) for x in line.strip()[len("RESULTS:"):].split(",") if x]
    bob_log.close()

    assert results is not None, f"Bob produced no RESULTS line; see {log_path}"
    return results


def test_ot_choice_zero_recovers_s0(backend):
    pairs, choices = [("0000", "1111")], [0]
    assert _run_ot_batch(pairs, choices, ELL) == _expected(pairs, choices)  # [0]


def test_ot_choice_one_recovers_s1(backend):
    pairs, choices = [("0000", "1111")], [1]
    assert _run_ot_batch(pairs, choices, ELL) == _expected(pairs, choices)  # [15]


def test_batch_of_ots_recovers_each_choice(backend):
    pairs = [("0000", "1111"), ("1010", "0101"), ("1100", "0011")]
    choices = [1, 0, 1]
    assert _run_ot_batch(pairs, choices, ELL) == _expected(pairs, choices)
