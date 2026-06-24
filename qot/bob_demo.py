import sys
from asyncio import StreamReader, StreamWriter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simulaqron.general.host_config import SocketsConfig
from simulaqron.sdk.protocol import SimulaQronClassicalServer
from simulaqron.settings import network_config, simulaqron_settings
from simulaqron.settings.network_config import NodeConfigType
 
from netqasm.runtime.settings import set_simulator
set_simulator("simulaqron")
 
from netqasm.sdk.external import NetQASMConnection  # noqa: E402
from netqasm.sdk import EPRSocket                    # noqa: E402
 
 
STATE_RUNNING_QUANTUM  = "RUNNING_QUANTUM"
STATE_WAITING_BASES    = "WAITING_BASES"
STATE_WAITING_MASKED   = "WAITING_MASKED"
STATE_DONE             = "DONE"
 
 
async def handle_quantum_bob(
    ctx: SimpleNamespace,
    y: int,
    ell: int,
    reader: StreamReader,
    writer: StreamWriter,
) -> str:
    ctx.x_tilde = np.zeros(4*ell, dtype=int)
    # sample bob's random measurement bases
    ctx.theta_tilde = np.random.randint(0, 2, size=4*ell).astype(int)

    # open a fresh connection and epr socket for this single OT
    epr_socket = EPRSocket("Alice")
    with NetQASMConnection("Bob", epr_sockets=[epr_socket], max_qubits=4*ell) as sim_conn:
        for j in range(4*ell):
            print(f"Bob: receiving qubit {j}", flush=True)
            # receive one half of the epr pair alice created
            epr_half = epr_socket.recv_keep(number=1)[0]
            sim_conn.flush()

            # receive and apply teleportation corrections from alice
            data = await reader.readline()
            if not data:
                print(f"Bob [RUNNING_QUANTUM]: connection dropped unexpectedly.", flush=True)
                break
            
            raw_msg = data.decode().strip()
            print(f"Bob [RUNNING_QUANTUM]: received '{raw_msg}'", flush=True)

            m1_str, m2_str = raw_msg.split(":")
            m1, m2 = int(m1_str), int(m2_str)
            if m2 == 1:
                epr_half.X()
            if m1 == 1:
                epr_half.Z()
            sim_conn.flush()

            # measure in bob's chosen basis
            if ctx.theta_tilde[j] == 1:
                epr_half.H()

            m = epr_half.measure()
            sim_conn.flush()

            ctx.x_tilde[j] = int(m)
    
    # signal alice that all qubits have been measured
    writer.write("MEASURED\n".encode())
    await writer.drain()
    return STATE_WAITING_BASES
 
 
async def handle_bases_bob(
    ctx: SimpleNamespace,
    y: int,
    ell: int,
    writer: StreamWriter,
    raw_msg: str,
) -> str:
    # receive alice's bases to find out where bob measured correctly
    theta = np.array([int(b) for b in raw_msg.split(",")], dtype=int)
    ctx.theta = theta

    # positions where bases matched: x_tilde == alice's x (bob can unmask s_y here)
    I_known    = np.where(ctx.theta == ctx.theta_tilde)[0]
    # positions where bases differed: x_tilde is uncorrelated (cannot unmask s_{1-y})
    I_unknown  = np.where(ctx.theta != ctx.theta_tilde)[0]

    # abort if either set is too small to extract ell indices
    if len(I_known) < ell or len(I_unknown) < ell:
        writer.write(b"ABORT\n")
        await writer.drain()
        return STATE_RUNNING_QUANTUM

    # take the first ell indices from each set
    recover_idx = I_known[:ell]
    hidden_idx  = I_unknown[:ell]
    # store recover_idx so handle_masked_bob can use it at unmask time
    ctx.recover_idx = recover_idx

    # assign index sets based on choice bit: bob puts recover_idx at position y
    if y == 0:
        I0, I1 = recover_idx, hidden_idx
    else:
        I0, I1 = hidden_idx, recover_idx

    I0_str = ",".join(map(str, I0))
    I1_str = ",".join(map(str, I1))

    writer.write(f"{I0_str}|{I1_str}\n".encode())
    await writer.drain()

    return STATE_WAITING_MASKED

 
async def handle_masked_bob(
    ctx: SimpleNamespace,
    y: int,
    writer: StreamWriter,
    raw_msg: str,
) -> str:
    # parse alice's masked strings t0 = s0^x[I0] and t1 = s1^x[I1]
    t0_str, t1_str = raw_msg.split("|")
    t0 = np.array([int(b) for b in t0_str.split(",") if b], dtype=int)
    t1 = np.array([int(b) for b in t1_str.split(",") if b], dtype=int)

    # unmask s_y: at recover_idx, x_tilde == alice's x so the mask cancels
    t_y = t0 if y == 0 else t1
    ctx.s_y = t_y ^ ctx.x_tilde[ctx.recover_idx]
    str_s_y = "".join(map(str, ctx.s_y))

    print(f"Bob: recovered s_{y} = {str_s_y}", flush=True)

    return STATE_DONE
 
 
def make_run_bob(y: int, ell: int):

    async def run_bob(reader: StreamReader, writer: StreamWriter) -> None:
        print("Bob: Alice connected.", flush=True)
        ctx = SimpleNamespace(
            theta_tilde=None,
            x_tilde=None,
            theta=None,
            recover_idx=None,
            s_y=None,
        )
 
        state = STATE_RUNNING_QUANTUM
        while state != STATE_DONE:
 
            if state == STATE_RUNNING_QUANTUM:
                state = await handle_quantum_bob(ctx, y, ell, reader, writer)
                continue
 
            data = await reader.readline()
            if not data:
                print(f"Bob [{state}]: connection dropped unexpectedly.", flush=True)
                break
            raw_msg = data.decode().strip()
            print(f"Bob [{state}]: received '{raw_msg}'", flush=True)
 
            if state == STATE_WAITING_BASES:
                state = await handle_bases_bob(ctx, y, ell, writer, raw_msg)
            elif state == STATE_WAITING_MASKED:
                state = await handle_masked_bob(ctx, y, writer, raw_msg)
 
        print(f"Bob: OT complete (final state: {state}).", flush=True)
        return ctx.s_y
 
    return run_bob
 
 
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 bob.py <ell> <y>")
        sys.exit(1)
 
    ell = int(sys.argv[1])
    y   = int(sys.argv[2])
 
    _here = Path(__file__).parent
    simulaqron_settings.read_from_file(_here / "simulaqron_settings.json")
    network_config.read_from_file(_here / "simulaqron_network.json")
 
    sockets_config = SocketsConfig(network_config, "default", NodeConfigType.APP)
    server = SimulaQronClassicalServer(sockets_config, "Bob")
    server.register_client_handler(make_run_bob(y, ell))
 
    print(f"Bob: starting OT server (ell={ell}, y={y})...", flush=True)
    server.start_serving()