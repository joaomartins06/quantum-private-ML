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
    """run a single OT over the already-open connection. retries on abort. based on the textbook's Protocol 10."""
    while True:
        # --- STEP 2. ---
        x_tilde = np.zeros(4 * ell, dtype=int)
        theta_tilde = np.random.randint(0, 2, size=4 * ell).astype(int)

        # quantum phase: receive and measure 4*ell qubits in bob's random bases
        for j in range(4 * ell):
            # receive one half of the epr pair alice created
            epr_half = epr_socket.recv_keep(number=1)[0]
            sim_conn.flush()

            # receive teleportation corrections from alice and apply them
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

            # measure in bob's chosen basis
            if theta_tilde[j] == 1:
                epr_half.H()
            m = epr_half.measure()
            sim_conn.flush()
            x_tilde[j] = int(m)

        print(f"Bob: quantum phase done ({4*ell} qubits received)", flush=True)

        # signal alice that all qubits have been measured
        writer.write(b"MEASURED\n")
        await writer.drain()

        # --- STEP 3. ---
        # receive alice's bases to find out where bob measured correctly
        data = await reader.readline()
        theta = np.array([int(b) for b in data.decode().strip().split(",")], dtype=int)

        # --- STEP 4. ---
        # positions where bases matched: x_tilde == alice's x (bob can unmask s_y here)
        I_known   = np.where(theta == theta_tilde)[0]
        # positions where bases differed: x_tilde is uncorrelated (cannot unmask s_{1-y})
        I_unknown = np.where(theta != theta_tilde)[0]

        print(f"Bob: matched {len(I_known)}/{4*ell} bases (need {ell} in each set)", flush=True)

        # abort if either set is too small to extract ell indices
        if len(I_known) < ell or len(I_unknown) < ell:
            print(f"Bob: not enough matches, sending ABORT", flush=True)
            writer.write(b"ABORT\n")
            await writer.drain()
            continue

        # take the first ell indices from each set
        recover_idx = I_known[:ell]
        hidden_idx  = I_unknown[:ell]

        # assign index sets based on choice bit: bob puts recover_idx at position y
        if y == 0:
            I0, I1 = recover_idx, hidden_idx
        else:
            I0, I1 = hidden_idx, recover_idx

        print(f"Bob: sending partition (y={y})", flush=True)
        writer.write(f"{','.join(map(str, I0))}|{','.join(map(str, I1))}\n".encode())
        await writer.drain()

        # --- STEP 6. ---
        # receive alice's masked strings t0 = s0^x[I0] and t1 = s1^x[I1]
        data = await reader.readline()
        t0_str, t1_str = data.decode().strip().split("|")
        t0 = np.array([int(b) for b in t0_str.split(",") if b], dtype=int)
        t1 = np.array([int(b) for b in t1_str.split(",") if b], dtype=int)

        # unmask s_y: at recover_idx, x_tilde == alice's x so the mask cancels
        t_y = t0 if y == 0 else t1
        result = t_y ^ x_tilde[recover_idx]

        print(f"Bob: recovered s_{y} = {bits_to_int(result)}", flush=True)
        return result


def make_run_bob(y_list, ell):
    async def run_bob(reader: StreamReader, writer: StreamWriter):
        print("Bob: Alice connected.", flush=True)
        results = []

        # open one connection and one epr socket for the entire batch
        epr_socket = EPRSocket("Alice")
        with NetQASMConnection("Bob", epr_sockets=[epr_socket],
                               max_qubits=4 * ell) as sim_conn:
            for k, y in enumerate(y_list):
                print(f"Bob: starting OT {k+1}/{len(y_list)}", flush=True)
                s_y = await one_ot_bob(sim_conn, epr_socket, int(y), ell, reader, writer)
                results.append(bits_to_int(s_y))
                print(f"Bob: OT {k + 1}/{len(y_list)} done", flush=True)

        # print all recovered values for the orchestrator to parse
        print("RESULTS:" + ",".join(map(str, results)), flush=True)
        return results

    return run_bob


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 bob_ot.py <ell> <y0,y1,...>")
        sys.exit(1)

    ell = int(sys.argv[1])
    # parse comma-separated choice bits from the command line
    y_list = [int(x) for x in sys.argv[2].split(",") if x != ""]

    _here = Path(__file__).parent
    simulaqron_settings.read_from_file(_here / "simulaqron_settings.json")
    network_config.read_from_file(_here / "simulaqron_network.json")

    sockets_config = SocketsConfig(network_config, "default", NodeConfigType.APP)
    server = SimulaQronClassicalServer(sockets_config, "Bob")
    server.register_client_handler(make_run_bob(y_list, ell))

    print(f"Bob: starting OT server (ell={ell}, {len(y_list)} OTs)...", flush=True)
    server.start_serving()