"""
Unit tests for the MPC primitives in ``multiply.py`` (Layer 2).

These are the algebraic core of the system: fixed-point encoding, additive
secret sharing, and Beaver-triple multiplication. They are pure, deterministic,
and need no quantum backend, so they run instantly:

    pytest mpc/test_multiply.py -v
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc.multiply import (
    encode, decode, share, reconstruct, beaver_mul, truncate,
)

ELL = 64
F = 16
MOD = 1 << ELL


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic shares so failures are reproducible."""
    random.seed(0)


def make_shared(x):
    """Additive shares of x mod 2^ELL."""
    return share(x % MOD, ELL)


def make_triple(a, b):
    """A valid Beaver triple for factors a, b: shares of (a, b, a*b)."""
    a0, a1 = share(a % MOD, ELL)
    b0, b1 = share(b % MOD, ELL)
    c0, c1 = share((a * b) % MOD, ELL)
    return ((a0, b0, c0), (a1, b1, c1))


# --- encode / decode (fixed-point round-trip) ------------------------------

@pytest.mark.parametrize("x", [0.0, 1.0, 3.14159, 0.001, 123.456])
def test_encode_decode_roundtrip_positive(x):
    assert decode(encode(x, F) % MOD, F, ELL) == pytest.approx(x, abs=1e-4)


@pytest.mark.parametrize("x", [-1.0, -3.14159, -0.5, -123.456])
def test_encode_decode_roundtrip_negative(x):
    # exercises the x >= mod//2 sign branch in decode
    assert decode(encode(x, F) % MOD, F, ELL) == pytest.approx(x, abs=1e-4)


# --- share / reconstruct ---------------------------------------------------

@pytest.mark.parametrize("x", [0, 1, 42, MOD - 1, 2 ** 40])
def test_share_reconstruct_roundtrip(x):
    s0, s1 = share(x, ELL)
    assert reconstruct(s0, s1, ELL) == x


def test_share_hides_secret():
    # neither share alone equals the secret (sanity: it really is split)
    x = 123456789
    s0, s1 = share(x, ELL)
    assert s0 != x and s1 != x
    assert (s0 + s1) % MOD == x


# --- beaver_mul ------------------------------------------------------------

@pytest.mark.parametrize("x,y", [
    (0, 0), (1, 1), (3, 7), (12345, 6789), (-5, 9), (-11, -13),
])
def test_beaver_mul_reconstructs_product(x, y):
    x_sh = make_shared(x)
    y_sh = make_shared(y)
    triple = make_triple(x, y)  # triple factors are independent of x, y in general
    c_sh = beaver_mul(x_sh, y_sh, triple, ELL)
    assert reconstruct(c_sh[0], c_sh[1], ELL) == (x * y) % MOD


def test_beaver_mul_triple_independent_of_operands():
    # the whole point of Beaver triples: the triple is unrelated to x, y
    x, y = 314, 159
    x_sh = make_shared(x)
    y_sh = make_shared(y)
    triple = make_triple(271828, 141421)  # arbitrary unrelated factors
    c_sh = beaver_mul(x_sh, y_sh, triple, ELL)
    assert reconstruct(c_sh[0], c_sh[1], ELL) == (x * y) % MOD


# --- truncate (SecureML local truncation, ±1 LSB error) --------------------

@pytest.mark.parametrize("value", [0, 1 << F, 5 << F, (123 << F) + 7])
def test_truncate_positive(value):
    z_sh = make_shared(value)
    t0, t1 = truncate(z_sh, F, ELL)
    got = reconstruct(t0, t1, ELL)
    assert abs(got - (value >> F)) <= 1


@pytest.mark.parametrize("value", [-(1 << F), -(5 << F), -((123 << F) + 7)])
def test_truncate_negative(value):
    z_sh = make_shared(value)
    t0, t1 = truncate(z_sh, F, ELL)
    got = decode(reconstruct(t0, t1, ELL), 0, ELL)  # interpret as signed int
    assert abs(got - (value / (1 << F))) <= 1
