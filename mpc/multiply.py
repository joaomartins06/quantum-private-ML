import numpy as np
import json
from pathlib import Path
from collections import deque
import random as _random



def encode(x, f): 
    return int(round(x * (1 << f)))


def decode(x, f, ell): 
    mod = 1 << ell
    #check if x >= 2^(ell-1) and if so, subtract 2^ell to get the negative value
    if x >= mod // 2:
        x -= mod
    return float(x / (1 << f))


def share(x, ell):
    mod = 1 << ell
    s0 = _random.randrange(mod)
    s1 = (x - s0) % mod
    return s0, s1


def reconstruct(s0, s1, ell):
    mod = 1 << ell
    return (s0 + s1) % mod


def load_triples(path):
    data = json.load(open(path))
    return deque(data)  # pop from left = consume in order


def next_triple(alice_q, bob_q):
    a = alice_q.popleft()
    b = bob_q.popleft()
    return ((a["u"], a["v"], a["z"]), (b["u"], b["v"], b["z"]))


def beaver_mul(x_sh, y_sh, triple, ell):
    mod = 1 << ell
    #e = (x0-a0),(x1-a1)
    e_shared_secret = ((x_sh[0] - triple[0][0]) % mod, (x_sh[1] - triple[1][0]) % mod)
    #f = (y0-b0),(y1-b1)
    f_shared_secret = ((y_sh[0] - triple[0][1]) % mod, (y_sh[1] - triple[1][1]) % mod)

    #I could sum rightaway, but, realistically, this would be done in a distributed way
    e = reconstruct(e_shared_secret[0], e_shared_secret[1], ell)
    f = reconstruct(f_shared_secret[0], f_shared_secret[1], ell)

    #c_i = -i * e * f + f * x_i + e * y_i + z_i 
    c = ((f * x_sh[0] + e * y_sh[0] + triple[0][2]) % mod, (-e * f + f * x_sh[1] + e * y_sh[1] + triple[1][2]) % mod)
    return c


def truncate(z_sh, f, ell):
    mod = 1 << ell
    z0, z1 = z_sh
    z0_trunc = z0 >> f
    z1_trunc = (mod - ((mod - z1) % mod >> f)) % mod
    return z0_trunc, z1_trunc



