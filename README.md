# Quantum-Private ML
 
**CS4090 — Quantum Cryptography (TU Delft)**  
Final project: quantum-secure private machine learning via BB84-based Oblivious Transfer.
 
---
 
## Overview
 
This project implements a privacy-preserving linear regression system where two parties collaboratively train a model on **vertically partitioned** data without either party ever seeing the other's raw features.
 
The system is built in three layers:

```
Layer 1 (qot/)   — Quantum Oblivious Transfer via BB84 + SimulaQron
Layer 2 (mpc/)   — Beaver triple generation (quantum or classical)
Layer 3 (ml/)    — Secure SGD linear regression over secret shares
```
 
The main paper is **SecureML** (Mohassel & Zhang, IEEE S&P 2017). The quantum primitive is **Protocol 10** from the CS4090 course book (BB84-based OT).
 
---
 
## Problem Setting
 
Two parties, Alice and Bob, each hold a disjoint subset of features for the same set of samples (vertical partition). They want to jointly train a linear regression model without either party revealing their raw data to the other. The only information exchanged at the end of training is the final weight vector `w`, reconstructed from additive shares held by both parties.
 
---
 
## Architecture
 
### Layer 1 — Quantum OT (`qot/`)
 
Implements BB84-based 1-out-of-2 Oblivious Transfer using SimulaQron. Alice holds two strings `s_0, s_1`; Bob holds a choice bit `y` and recovers `s_y` without Alice learning `y` or Bob learning `s_{1-y}`.
 
- `alice_ot.py`, `bob_ot.py` — batched OT over a single persistent `NetQASMConnection` per party.
- `alice_demo.py`, `bob_demo.py` — single-OT scripts for isolated testing.
### Layer 2 — Beaver Triple Generation (`mpc/`)
 
Beaver triples `(a, b, c)` with `c = a·b mod 2^ell` are the offline cryptographic resource consumed by every secure multiplication during training. Two sources are provided:
 
- `triple.py` + `batch_triples_q.py` — quantum-generated triples via Layer 1 (cryptographically secure, slow).
- `triples_generator.py` — classical triples generated directly (no privacy claim, instant; used for Layer 3 development).
Triples are cached to `triples/` (quantum) and `triples_classical/` (classical) as `alice.json` / `bob.json`.
 
- `multiply.py` — shared MPC primitives: `encode`, `decode`, `share`, `reconstruct`, `beaver_mul`, `truncate`.
### Layer 3 — Secure Linear Regression (`ml/`)
 
Mini-batch SGD where all cross-party multiplications go through Beaver triples. Fixed-point encoding at `ell=64`, `f=16` fractional bits. Per-batch triple consumption: `2 × B × d` triples (forward + backward pass).
 
- `train.py` — full training pipeline: data encoding, secure forward/backward, weight update, final reconstruction.
---
 
## Demo
 
The full pipeline has been applied to the **sklearn diabetes dataset** (442 patients, 10 features, continuous target), with Alice holding features 0–4 and Bob holding features 5–9. The secure model converges correctly and matches the accuracy of a plaintext SGD baseline, demonstrating negligible accuracy loss from the privacy-preserving protocol.
 
---
 
## Setup
 
```bash
python3 -m venv simulaqron-venv
source simulaqron-venv/bin/activate
pip install simulaqron netqasm scikit-learn numpy matplotlib
```
 
### Generate classical triples
 
```bash
# Edit N and ELL in triples_generator.py (N=100000, ELL=64 recommended)
python mpc/triples_generator.py
```
 
### Generate quantum triples
 
```bash
# From repo root
simulaqron start --nodes Alice,Bob \
  --network-config-file qot/simulaqron_network.json
 
python mpc/batch_triples_q.py
 
simulaqron stop
```
 
### Run secure training
 
```bash
python ml/train.py
```
 
### Cleanup after interrupted runs
 
```bash
simulaqron stop
pkill -9 -f simulaqron
pkill -9 -f alice_ot
pkill -9 -f bob_ot
rm -f ~/.simulaqron_pids/*
```
 
---
 
## References
 
- Mohassel, P. & Zhang, Y. (2017). *SecureML: A System for Scalable Privacy-Preserving Machine Learning.* IEEE S&P 2017.
- Vidick, T. & Wehner, S. *Introduction to Quantum Cryptography*. Cambridge University Press. Protocol 10 (BB84-based OT).
- SimulaQron: http://www.simulaqron.org