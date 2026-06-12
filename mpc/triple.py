import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import time
import subprocess


_here = Path(__file__).parent.parent / "qot"


def int_to_bits(n, ell):
    return np.array([(n >> i) & 1 for i in range(ell)], dtype=int)


def bits_to_int(bits):
    return int(bits @ (1 << np.arange(len(bits))))


def _restart_backend():
    subprocess.run(["simulaqron", "stop"], capture_output=True)
    time.sleep(2.0)
    subprocess.Popen(
        ["simulaqron", "start", "--nodes", "Alice,Bob",
         "--network-config-file", "qot/simulaqron_network.json"],
        cwd=_here.parent
    )
    time.sleep(5.0)


def run_one_ot(s0, s1, y, ell):
    _restart_backend()

    s0_str = "".join(map(str, s0))
    s1_str = "".join(map(str, s1))

    bob_proc = subprocess.Popen(
        ["python", "bob_ot.py", str(ell), str(y)],
        cwd=_here, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    time.sleep(2.0)

    alice_result = subprocess.run(
        ["python", "alice_ot.py", str(ell), s0_str, s1_str],
        cwd=_here, capture_output=True, text=True, timeout=60
    )

    if alice_result.returncode != 0:
        bob_proc.kill()
        bob_proc.wait()
        raise RuntimeError(f"Alice failed: {alice_result.stderr}")

    # Bob's stdout may contain log lines before the result; find the bitstring line
    s_y_line = None
    while True:
        line = bob_proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line and all(c in "01" for c in line):
            s_y_line = line
            break

    bob_proc.terminate()
    try:
        bob_proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        bob_proc.kill()
        bob_proc.wait()

    if s_y_line is None:
        bob_stderr = bob_proc.stderr.read()
        raise RuntimeError(f"Bob produced no bitstring output. stderr:\n{bob_stderr}")

    return np.array([int(b) for b in s_y_line], dtype=int)


def generate_triple(ell):
    mod = 2**ell

    u0 = np.random.randint(0, mod)
    v0 = np.random.randint(0, mod)
    u1 = np.random.randint(0, mod)
    v1 = np.random.randint(0, mod)

    cross_A_bob = 0
    cross_A_alice = 0
    cross_B_bob = 0
    cross_B_alice = 0

    for k in range(ell):
        r_k = np.random.randint(0, mod)
        s0 = int_to_bits(r_k, ell)
        s1 = int_to_bits((u0 * (1 << k) + r_k) % mod, ell)
        result = run_one_ot(s0, s1, int(int_to_bits(v1, ell)[k]), ell)
        cross_A_bob = (cross_A_bob + bits_to_int(result)) % mod
        cross_A_alice = (cross_A_alice - r_k) % mod
        print(f'OT {2*k+1}/{2*ell} done')

        r_k = np.random.randint(0, mod)
        s0 = int_to_bits(r_k, ell)
        s1 = int_to_bits((v0 * (1 << k) + r_k) % mod, ell)
        result = run_one_ot(s0, s1, int(int_to_bits(u1, ell)[k]), ell)
        cross_B_bob = (cross_B_bob + bits_to_int(result)) % mod
        cross_B_alice = (cross_B_alice - r_k) % mod
        print(f'OT {2*k+2}/{2*ell} done')

    z0 = (u0 * v0 + cross_A_alice + cross_B_alice) % mod
    z1 = (u1 * v1 + cross_A_bob + cross_B_bob) % mod

    return (u0, v0, z0), (u1, v1, z1)


if __name__ == "__main__":
    ell = 4
    (u0, v0, z0), (u1, v1, z1) = generate_triple(ell)
    print(f"Alice: u0={u0}, v0={v0}, z0={z0}")
    print(f"Bob:   u1={u1}, v1={v1}, z1={z1}")

    mod = 2**ell
    check = ((u0+u1) % mod) * ((v0+v1) % mod) % mod
    total = (z0 + z1) % mod
    print(f"Check: (u0+u1)(v0+v1) mod {mod} = {check}, z0+z1 mod {mod} = {total}, match={check==total}")

    triples_dir = Path(__file__).parent / "triples"
    triples_dir.mkdir(exist_ok=True)
    np.savez(triples_dir / "alice.npz", u=u0, v=v0, z=z0, ell=ell)
    np.savez(triples_dir / "bob.npz", u=u1, v=v1, z=z1, ell=ell)
    print(f"Saved to {triples_dir}/alice.npz and bob.npz")

