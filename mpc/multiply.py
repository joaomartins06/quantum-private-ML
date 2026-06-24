import numpy as np
import json
from pathlib import Path
from collections import deque
import random as _random


def encode(x, f):
    # scale x by 2^f and round to the nearest integer
    return int(round(x * (1 << f)))


def decode(x, f, ell):
    mod = 1 << ell
    # two's complement, values above 2^(ell-1) are negative
    if x >= mod // 2:
        x -= mod
    return float(x / (1 << f))


def share(x, ell):
    mod = 1 << ell
    # pick s0 at random, compute s1 so that s0 + s1 = x mod 2^ell
    s0 = _random.randrange(mod)
    s1 = (x - s0) % mod
    return s0, s1


def reconstruct(s0, s1, ell):
    mod = 1 << ell
    return (s0 + s1) % mod


def load_triples(path):
    data = json.load(open(path))
    # consume in order, never reuse
    return deque(data)


def next_triple(alice_q, bob_q):
    a = alice_q.popleft()
    b = bob_q.popleft()
    return ((a["u"], a["v"], a["z"]), (b["u"], b["v"], b["z"]))


def beaver_mul(x_sh, y_sh, triple, ell):
    mod = 1 << ell
    # mask x and y with the triple's a, b components
    # e = x - a, f = y - b, both still in shares
    e_shared_secret = ((x_sh[0] - triple[0][0]) % mod, (x_sh[1] - triple[1][0]) % mod)
    f_shared_secret = ((y_sh[0] - triple[0][1]) % mod, (y_sh[1] - triple[1][1]) % mod)

    # open e and f, in a real deployment each party sends its share to the other
    e = reconstruct(e_shared_secret[0], e_shared_secret[1], ell)
    f = reconstruct(f_shared_secret[0], f_shared_secret[1], ell)

    # compute output shares using the formula c_i = f*x_i + e*y_i + z_i - i*e*f
    # party 0 adds the public e*f term, party 1 subtracts it
    c = ((f * x_sh[0] + e * y_sh[0] + triple[0][2]) % mod,
         (-e * f + f * x_sh[1] + e * y_sh[1] + triple[1][2]) % mod)
    return c


def truncate(z_sh, f, ell):
    mod = 1 << ell
    z0, z1 = z_sh
    # party 0 right-shifts directly
    z0_trunc = z0 >> f
    # party 1 needs a sign-aware correction to preserve the additive invariant
    z1_trunc = (mod - ((mod - z1) % mod >> f)) % mod
    return z0_trunc, z1_trunc