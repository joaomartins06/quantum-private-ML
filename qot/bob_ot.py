import sys
from asyncio import StreamReader, StreamWriter
from pathlib import Path

import numpy as np

from simulaqron.general.host_config import SocketsConfig
from simulaqron.sdk.protocol import SimulaQronClassicalServer
from simulaqron.settings import network_config, simulaqron_settings
from simulaqron.settings.network_config import NodeConfigType

from netqasm.runtime.settings import set_simulator
set_simulator("simulaqron")

from netqasm.sdk.external import NetQASMConnection  # noqa: E402
from netqasm.sdk import EPRSocket                    # noqa: E402


def bits_to_int(bits):
    return int(bits @ (1 << np.arange(len(bits))))


async def one_ot_bob(sim_conn, epr_socket, y, ell, reader, writer):
    """Run a single OT over the already-open connection. Retries on abort."""
    while True:
        x_tilde = np.zeros(4 * ell, dtype=int)
        theta_tilde = np.random.randint(0, 2, size=4 * ell).astype(int)

        #quantum phase: receive + BB84-measure 4*ell qubits
        for j in range(4 * ell):
            epr_half = epr_socket.recv_keep(number=1)[0]
            sim_conn.flush()

            data = await reader.readline()
            if not data:
                raise RuntimeError("Bob: connection dropped during quantum phase")
            m1_str, m2_str = data.decode().strip().split(":")
            m1, m2 = int(m1_str), int(m2_str)
            if m2 == 1:
                epr_half.X()
            if m1 == 1:
                epr_half.Z()
            sim_conn.flush()

            if theta_tilde[j] == 1:
                epr_half.H()
            m = epr_half.measure()
            sim_conn.flush()
            x_tilde[j] = int(m)

        writer.write(b"MEASURED\n")
        await writer.drain()

        #receive theta, compute matching set
        data = await reader.readline()
        theta = np.array([int(b) for b in data.decode().strip().split(",")], dtype=int)
        I = np.where(theta == theta_tilde)[0]

        if len(I) < 2 * ell:
            writer.write(b"ABORT\n")
            await writer.drain()
            continue  # retry this OT only

        I_y = I[:ell]
        I_1y = I[ell:2 * ell]
        if y == 0:
            I0, I1 = I_y, I_1y
        else:
            I0, I1 = I_1y, I_y

        writer.write(f"{','.join(map(str, I0))}|{','.join(map(str, I1))}\n".encode())
        await writer.drain()

        #receive masked messages, recover s_y
        data = await reader.readline()
        t0_str, t1_str = data.decode().strip().split("|")
        t0 = np.array([int(b) for b in t0_str.split(",") if b], dtype=int)
        t1 = np.array([int(b) for b in t1_str.split(",") if b], dtype=int)

        if y == 0:
            t_y, idx = t0, I0
        else:
            t_y, idx = t1, I1

        return t_y ^ x_tilde[idx]


def make_run_bob(y_list, ell):
    async def run_bob(reader: StreamReader, writer: StreamWriter):
        print("Bob: Alice connected.", flush=True)
        results = []

        epr_socket = EPRSocket("Alice")                       # ONCE
        with NetQASMConnection("Bob", epr_sockets=[epr_socket],
                               max_qubits=4 * ell) as sim_conn:  #ONCE, wraps batch
            for k, y in enumerate(y_list):
                s_y = await one_ot_bob(sim_conn, epr_socket, int(y), ell, reader, writer)
                results.append(bits_to_int(s_y))
                print(f"Bob: OT {k + 1}/{len(y_list)} done", flush=True)

        print("RESULTS:" + ",".join(map(str, results)), flush=True)
        return results

    return run_bob


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 bob_ot.py <ell> <y0,y1,...>")
        sys.exit(1)

    ell = int(sys.argv[1])
    y_list = [int(x) for x in sys.argv[2].split(",") if x != ""]

    _here = Path(__file__).parent
    simulaqron_settings.read_from_file(_here / "simulaqron_settings.json")
    network_config.read_from_file(_here / "simulaqron_network.json")

    sockets_config = SocketsConfig(network_config, "default", NodeConfigType.APP)
    server = SimulaQronClassicalServer(sockets_config, "Bob")
    server.register_client_handler(make_run_bob(y_list, ell))

    print(f"Bob: starting OT server (ell={ell}, {len(y_list)} OTs)...", flush=True)
    server.start_serving()