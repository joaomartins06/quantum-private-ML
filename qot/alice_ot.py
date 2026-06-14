import sys
from asyncio import StreamReader, StreamWriter
from pathlib import Path

import numpy as np

from simulaqron.general.host_config import SocketsConfig
from simulaqron.sdk.protocol import SimulaQronClassicalClient
from simulaqron.settings import network_config, simulaqron_settings
from simulaqron.settings.network_config import NodeConfigType

from netqasm.runtime.settings import set_simulator
set_simulator("simulaqron")

from netqasm.sdk.external import NetQASMConnection  # noqa: E402
from netqasm.sdk import Qubit, EPRSocket            # noqa: E402


async def one_ot_alice(sim_conn, epr_socket, s0, s1, ell, reader, writer):
    while True:
        x = np.random.randint(0, 2, size=4 * ell).astype(int)
        theta = np.random.randint(0, 2, size=4 * ell).astype(int)

        # quantum phase: teleport 4*ell BB84 qubits 
        for i in range(4 * ell):
            epr_half = epr_socket.create_keep(number=1)[0]
            q = Qubit(sim_conn)
            if x[i] == 1:
                q.X()
            if theta[i] == 1:
                q.H()
            q.cnot(epr_half)
            q.H()
            m1 = q.measure()
            m2 = epr_half.measure()
            sim_conn.flush()
            writer.write(f"{int(m1)}:{int(m2)}\n".encode())
            await writer.drain()

        # wait for MEASURED, then send theta
        data = await reader.readline()
        if data.decode().strip() != "MEASURED":
            raise RuntimeError(f"Alice: expected MEASURED, got '{data.decode().strip()}'")
        writer.write((",".join(map(str, theta)) + "\n").encode())
        await writer.drain()

        #receive partition (or ABORT)
        data = await reader.readline()
        msg = data.decode().strip()
        if msg == "ABORT":
            continue  # retry this OT only

        i0, i1 = msg.split("|", 1)
        I0 = np.array([int(i) for i in i0.split(",") if i], dtype=int)
        I1 = np.array([int(i) for i in i1.split(",") if i], dtype=int)

        t0 = s0 ^ x[I0]
        t1 = s1 ^ x[I1]
        writer.write(f"{','.join(map(str, t0))}|{','.join(map(str, t1))}\n".encode())
        await writer.drain()
        return


def make_run_alice(s_list, ell):
    async def run_alice(reader: StreamReader, writer: StreamWriter) -> None:
        epr_socket = EPRSocket("Bob")                          # ONCE
        with NetQASMConnection("Alice", epr_sockets=[epr_socket],
                               max_qubits=4 * ell) as sim_conn:  # ONCE, wraps batch
            for k, (s0, s1) in enumerate(s_list):
                await one_ot_alice(sim_conn, epr_socket, s0, s1, ell, reader, writer)
                print(f"Alice: OT {k + 1}/{len(s_list)} done", flush=True)
        print("Alice: all OTs complete.", flush=True)

    return run_alice


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 alice_ot.py <ell> <s0,s1;s0,s1;...>")
        sys.exit(1)

    ell = int(sys.argv[1])
    s_list = []
    for pair in sys.argv[2].split(";"):
        s0_str, s1_str = pair.split(",")
        s_list.append((
            np.array([int(c) for c in s0_str], dtype=int),
            np.array([int(c) for c in s1_str], dtype=int),
        ))

    _here = Path(__file__).parent
    simulaqron_settings.read_from_file(_here / "simulaqron_settings.json")
    network_config.read_from_file(_here / "simulaqron_network.json")

    sockets_config = SocketsConfig(network_config, "default", NodeConfigType.APP)
    client = SimulaQronClassicalClient(sockets_config)

    print(f"Alice: starting OT (ell={ell}, {len(s_list)} OTs)", flush=True)
    client.run_client("Bob", make_run_alice(s_list, ell))