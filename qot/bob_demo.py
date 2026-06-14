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
    ctx.theta_tilde = np.random.randint(0, 2, size=4*ell).astype(int)

    epr_socket = EPRSocket("Alice")
    with NetQASMConnection("Bob", epr_sockets=[epr_socket], max_qubits=4*ell) as sim_conn:
        for j in range(4*ell):
            print(f"Bob: receiving qubit {j}", flush=True)
            epr_half = epr_socket.recv_keep(number=1)[0]
            sim_conn.flush()

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

            if ctx.theta_tilde[j] == 1:
                epr_half.H()

            m = epr_half.measure()
            sim_conn.flush()

            ctx.x_tilde[j] = int(m)
    
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

    theta = np.array([int(b) for b in raw_msg.split(",")], dtype=int)
    ctx.theta = theta

    I = np.where(ctx.theta == ctx.theta_tilde)[0]
    if len(I) < 2*ell:
        writer.write(b"ABORT\n")
        await writer.drain()
        return STATE_RUNNING_QUANTUM
    
    I_y    = I[:ell]
    I_1y   = I[ell:2*ell]
    
    if y == 0:
        I0, I1 = I_y, I_1y
    else:
        I0, I1 = I_1y, I_y

    ctx.I0 = I0
    ctx.I1 = I1

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


    t0_str, t1_str = raw_msg.split("|")
    t0 = np.array([int(b) for b in t0_str.split(",") if b], dtype=int)
    t1 = np.array([int(b) for b in t1_str.split(",") if b], dtype=int)

    if y == 0:
        t_y = t0
        I_y = ctx.I0
    else:
        t_y = t1
        I_y = ctx.I1

    ctx.s_y = t_y ^ ctx.x_tilde[I_y]
    str_s_y = "".join(map(str, ctx.s_y))

    print(f"Bob: recovered s_{y} = {str_s_y}", flush=True)

    return STATE_DONE
 
 
# ── Event loop ────────────────────────────────────────────────────────────────
 
def make_run_bob(y: int, ell: int):

    async def run_bob(reader: StreamReader, writer: StreamWriter) -> None:
        print("Bob: Alice connected.", flush=True)
        ctx = SimpleNamespace(
            theta_tilde=None,
            x_tilde=None,
            theta=None,
            I0=None,
            I1=None,
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
 
 
# ── Entry point (for isolated testing) ───────────────────────────────────────
 
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