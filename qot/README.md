# Layer 1 — Quantum Oblivious Transfer (`qot/`)

This is the cryptographic foundation of the project. It implements **1-out-of-2
Oblivious Transfer (OT)** on top of **BB84** using the **SimulaQron** quantum
network simulator. Everything in `mpc/` and `ml/` ultimately rests on the
security primitive defined here.

> **1-out-of-2 OT.** Alice holds two strings `s0` and `s1`. Bob holds a choice
> bit `y` and learns **only** `s_y`. Alice never learns `y`; Bob never learns
> `s_{1-y}`.

This README explains the protocol, maps it to the code, and — importantly —
shows how to run and **test Layer 1 in isolation**, without touching `mpc/` or
`ml/`.

---

## Files

| File | Role                                                                                                  |
|------|-------------------------------------------------------------------------------------------------------|
| `alice_ot.py` | Alice (OT **sender**), **batched**: many OTs over one persistent `NetQASMConnection`. Used by `mpc/`. |
| `bob_ot.py` | Bob (OT **receiver**), batched counterpart of `alice_ot.py`.                                          |
| `alice_demo.py` | Single-OT Alice, written as an explicit state machine. For reading and isolated testing.              |
| `bob_demo.py` | Single-OT Bob, state-machine counterpart of `alice_demo.py`.                                          |
| `simulaqron_network.json` | SimulaQron config — _directly taken from previous course assignments._                                |
| `simulaqron_settings.json` | _idem_                                                                                                |
| `test_quantum.py` | End-to-end `pytest` tests that run the real OT over a live SimulaQron backend.                        |

---

## The protocol (and where it lives in the code)

> Implementation of the textbook's Protocol 10.

The quantum transport only matters for one reason: it lets Bob learn Alice's
bits on a subset of positions of his choosing, while leaving him ignorant on the
rest. Everything else is classical post-processing.

For an OT of `ell`-bit strings, the parties use `4*ell` BB84 qubits (instead of the textbook's `2*ell`, too small for practical implementation).

1. **Quantum phase.** Alice picks random bits `x` and random bases `theta`
   (each `4*ell` long) and sends `|x⟩` in bases `theta` to Bob (here: prepared
   locally and teleported over an EPR pair). Bob measures in his own random
   bases `theta_tilde`, obtaining `x_tilde`.
   *BB84 guarantee:* where `theta == theta_tilde`, `x_tilde == x`; where they
   differ, `x_tilde` is uniformly random and uncorrelated with `x`.
   → `one_ot_alice` / `one_ot_bob` (batched), `handle_quantum_*` (demo).

2. **Reveal bases.** Bob signals `MEASURED`; Alice then sends `theta`.

3. **Partition.** Bob forms his chosen index set `I_y` from the **matched**
   positions `{ i : theta[i] == theta_tilde[i] }` (there `x_tilde == x`, so he
   can unmask `s_y`) and the other set `I_{1-y}` from the **mismatched**
   complement (there `x_tilde` is uncorrelated with `x`, so he *cannot* unmask
   `s_{1-y}` — this is what enforces sender privacy). If either set has fewer
   than `ell` positions he sends `ABORT` and the OT retries. He sends the two
   `ell`-sized sets as `(I0, I1)` — **without** revealing which is his choice,
   which hides `y` from Alice.
   → `handle_bases_bob` / partition block in `one_ot_bob`.

4. **Mask.** Alice replies with `t0 = s0 ^ x[I0]` and `t1 = s1 ^ x[I1]`.
   → tail of `one_ot_alice` / `handle_partition_alice`.

5. **Recover.** Bob computes `s_y = t_y ^ x_tilde[I_y]`, where `I_y` is the
   index set he aligned with his choice. Because that set is one where his bases
   matched, `x_tilde == x` there and the mask cancels.
   → `handle_masked_bob` / tail of `one_ot_bob`.

Classical messages travel over the SimulaQron classical socket
(`SimulaQronClassicalClient`/`Server`); the `m1:m2` lines in the quantum phase
are the teleportation corrections.

### Bit/integer convention

Bit strings are **little-endian**: `bits_to_int(b) == sum(b[i] * 2**i)`
(`qot/bob_ot.py`, `mpc/triple.py`). So `1010` decodes to `1 + 4 = 5`, not `10`.

---

## Running Layer 1 standalone

All commands assume the `simulaqron-venv` from the top-level README and are run
**from the repo root** unless noted.

### A. Automated end-to-end tests (require SimulaQron)

`test_quantum.py` runs the real OT over a live backend and asserts Bob recovers
`s_y` for each choice bit. Run it from the `simulaqron-venv`:

```bash
pytest qot/test_quantum.py -v
```

It starts and stops its own backend. These tests are slow (every qubit is
simulated). The whole file is **skipped automatically** if the `simulaqron` CLI
is not on `PATH`, so it stays green in environments without the quantum stack.

### B. Single OT, by hand (two terminals)

Start the backend once:

```bash
simulaqron start --nodes Alice,Bob \
  --network-config-file qot/simulaqron_network.json
```

Terminal 1 — Bob (receiver), choice bit `y=1`, `ell=4`:

```bash
cd qot && python bob_demo.py 4 1
```

Terminal 2 — Alice (sender), `s0=0000`, `s1=1111`, `ell=4`:

```bash
cd qot && python alice_demo.py 4 0000 1111
```

Bob should print `recovered s_1 = 1111`. Run again with `y=0` and Bob recovers
`0000`. Stop the backend with `simulaqron stop`.

### C. Batched OT (the interface `mpc/` uses)

`alice_ot.py`/`bob_ot.py` run **many** OTs over a single connection. `s0;s1`
pairs are `;`-separated, choice bits are `,`-separated:

```bash
# Bob: two OTs, choices y = [1, 0]
cd qot && python bob_ot.py 4 1,0

# Alice: matching two (s0,s1) pairs
cd qot && python alice_ot.py 4 "0000,1111;1010,0101"
```

Bob prints a final `RESULTS:` line with the recovered integers
(`15, 5` here — `1111` → 15 and little-endian `1010` → 5). This is exactly what
`mpc/triple.py::run_batch_ot` parses.

---

## Troubleshooting

Stuck processes are the usual failure mode after an interrupted run:

```bash
simulaqron stop
pkill -9 -f simulaqron
pkill -9 -f alice_ot
pkill -9 -f bob_ot
rm -f ~/.simulaqron_pids/*
```

- **Port already in use** → a previous backend is alive; run the cleanup above.
- **Alice connects before Bob is serving** → start Bob first (the batched runner
  in `mpc/` sleeps ~2 s between the two for this reason).
- **It's extremely slow** → expected. SimulaQron simulates every qubit; this is
  why `mpc/triples_generator.py` (classical triples) exists for developing
  Layers 2–3.
