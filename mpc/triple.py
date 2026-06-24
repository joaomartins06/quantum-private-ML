import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import time
import subprocess


_qot = Path(__file__).parent.parent / "qot"
_repo = Path(__file__).parent.parent


def int_to_bits(n, ell):
    # little-endian: bit i has weight 2^i
    return np.array([(n >> i) & 1 for i in range(ell)], dtype=int)


def bits_to_int(bits):
    return int(bits @ (1 << np.arange(len(bits))))


def start_backend():
    # stop any leftover backend, wait for it to die, then start fresh
    subprocess.run(["simulaqron", "stop"], capture_output=True)
    time.sleep(3.0)
    subprocess.Popen(
        ["simulaqron", "start", "--nodes", "Alice,Bob",
         "--network-config-file", "qot/simulaqron_network.json"],
        cwd=_repo,
    )
    time.sleep(5.0)


def stop_backend():
    subprocess.run(["simulaqron", "stop"], capture_output=True)


def run_batch_ot(s_list, y_list, ell):
    # pack all (s0, s1) pairs and choice bits into command-line arguments
    y_arg = ",".join(str(int(y)) for y in y_list)
    s_arg = ";".join(
        f"{''.join(map(str, s0))},{''.join(map(str, s1))}" for (s0, s1) in s_list
    )

    # redirect bob's output to a log file so we can parse the RESULTS line later
    log_path = Path(__file__).parent / "bob_out.log"
    bob_log = open(log_path, "w+")
    bob_proc = subprocess.Popen(
        ["python", "bob_ot.py", str(ell), y_arg],
        cwd=_qot, stdout=bob_log, stderr=subprocess.STDOUT, text=True,
    )
    # give bob time to start serving before alice connects
    time.sleep(2.0)

    alice_result = subprocess.run(
        ["python", "alice_ot.py", str(ell), s_arg],
        cwd=_qot, capture_output=True, text=True, timeout=300,
    )
    # print alice's output so it appears in the main terminal
    print(alice_result.stdout, flush=True)

    if alice_result.returncode != 0:
        bob_proc.terminate()
        bob_proc.wait()
        bob_log.close()
        raise RuntimeError(f"Alice failed:\n{alice_result.stderr}")

    # give bob time to flush the RESULTS line before we terminate it
    time.sleep(1.0)
    bob_proc.terminate()
    try:
        bob_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        bob_proc.kill()
        bob_proc.wait()

    # parse bob's recovered values from the log
    bob_log.seek(0)
    results = None
    for line in bob_log:
        line = line.strip()
        if line.startswith("RESULTS:"):
            results = [int(x) for x in line[len("RESULTS:"):].split(",") if x != ""]
    bob_log.close()

    if results is None:
        raise RuntimeError(f"Bob produced no RESULTS line. See {log_path}")
    if len(results) != len(y_list):
        raise RuntimeError(f"Expected {len(y_list)} results, got {len(results)}")
    return results


def generate_triple(ell):
    mod = 2 ** ell

    # alice's shares of u and v
    u0 = np.random.randint(0, mod)
    v0 = np.random.randint(0, mod)
    # bob's shares of u and v
    u1 = np.random.randint(0, mod)
    v1 = np.random.randint(0, mod)

    # build the ot schedule: 2*ell OTs, interleaved between cross terms A and B
    # even index 2k   -> cross term A, bob selects with bit v1[k]
    # odd  index 2k+1 -> cross term B, bob selects with bit u1[k]
    s_list, y_list = [], []
    cross_A_alice = 0
    cross_B_alice = 0
    for k in range(ell):
        # cross term A: alice's OT inputs for bit k
        r = np.random.randint(0, mod)
        s_list.append((int_to_bits(r, ell), int_to_bits((u0 * (1 << k) + r) % mod, ell)))
        y_list.append(int(int_to_bits(v1, ell)[k]))
        cross_A_alice = (cross_A_alice - r) % mod

        # cross term B: alice's OT inputs for bit k
        r = np.random.randint(0, mod)
        s_list.append((int_to_bits(r, ell), int_to_bits((v0 * (1 << k) + r) % mod, ell)))
        y_list.append(int(int_to_bits(u1, ell)[k]))
        cross_B_alice = (cross_B_alice - r) % mod

    results = run_batch_ot(s_list, y_list, ell)

    # reconstruct cross terms from bob's recovered values
    cross_A_bob = sum(results[0::2]) % mod
    cross_B_bob = sum(results[1::2]) % mod

    # compute z shares: z = u*v = u0*v0 + u1*v1 + cross_A + cross_B
    z0 = (u0 * v0 + cross_A_alice + cross_B_alice) % mod
    z1 = (u1 * v1 + cross_A_bob + cross_B_bob) % mod

    return (u0, v0, z0), (u1, v1, z1)


if __name__ == "__main__":
    ell = 4
    start_backend()
    try:
        (u0, v0, z0), (u1, v1, z1) = generate_triple(ell)
    finally:
        stop_backend()

    print(f"Alice: u0={u0}, v0={v0}, z0={z0}")
    print(f"Bob:   u1={u1}, v1={v1}, z1={z1}")

    mod = 2 ** ell
    check = ((u0 + u1) % mod) * ((v0 + v1) % mod) % mod
    total = (z0 + z1) % mod
    print(f"Check: (u0+u1)(v0+v1) mod {mod} = {check}, "
          f"z0+z1 mod {mod} = {total}, match={check == total}")

    triples_dir = Path(__file__).parent / "triples"
    triples_dir.mkdir(exist_ok=True)
    np.savez(triples_dir / "alice.npz", u=u0, v=v0, z=z0, ell=ell)
    np.savez(triples_dir / "bob.npz", u=u1, v=v1, z=z1, ell=ell)
    print(f"Saved to {triples_dir}/alice.npz and bob.npz")