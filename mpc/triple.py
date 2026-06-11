import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from asyncio import StreamReader, StreamWriter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import threading


from simulaqron.general.host_config import SocketsConfig
from simulaqron.sdk.protocol import SimulaQronClassicalClient, SimulaQronClassicalServer
from simulaqron.settings import network_config, simulaqron_settings
from simulaqron.settings.network_config import NodeConfigType

from qot.alice import make_run_alice
from qot.bob import make_run_bob


_here = Path(__file__).parent.parent / "qot"

simulaqron_settings.read_from_file(_here / "simulaqron_settings.json")
network_config.read_from_file(_here / "simulaqron_network.json")

sockets_config = SocketsConfig(network_config, "default", NodeConfigType.APP)

client = SimulaQronClassicalClient(sockets_config)
server = SimulaQronClassicalServer(sockets_config, "Bob")


def int_to_bits(n, ell):
    return np.array([(n >> i) & 1 for i in range(ell)], dtype=int)


def bits_to_int(bits):
    return int(bits @ (1 << np.arange(len(bits))))


def run_one_ot(s0, s1, y, ell):
    run_alice = make_run_alice(s0, s1, ell)
    run_bob, result = make_run_bob(y, ell)
    
    server.register_client_handler(run_bob)
    threading.Thread(target=server.start_serving, daemon=True).start()
    client.run_client("Bob", run_alice)
    
    return result


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
        cross_A_bob = (cross_A_bob + bits_to_int(result.s_y)) % mod
        cross_A_alice = (cross_A_alice - r_k) % mod

        r_k = np.random.randint(0, mod)
        s0 = int_to_bits(r_k, ell)
        s1 = int_to_bits((v0 * (1 << k) + r_k) % mod, ell)
        result = run_one_ot(s0, s1, int(int_to_bits(u1, ell)[k]), ell)
        cross_B_bob = (cross_B_bob + bits_to_int(result.s_y)) % mod
        cross_B_alice = (cross_B_alice - r_k) % mod

    z0 = (u0 * v0 + cross_A_alice + cross_B_alice) % mod
    z1 = (u1 * v1 + cross_A_bob + cross_B_bob) % mod

    return (u0, v0, z0), (u1, v1, z1)

    

    





