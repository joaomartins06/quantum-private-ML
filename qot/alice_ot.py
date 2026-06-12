import sys
from asyncio import StreamReader, StreamWriter
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simulaqron.general.host_config import SocketsConfig
from simulaqron.sdk.protocol import SimulaQronClassicalClient
from simulaqron.settings import network_config, simulaqron_settings
from simulaqron.settings.network_config import NodeConfigType
 
from netqasm.runtime.settings import set_simulator
set_simulator("simulaqron")
 
from netqasm.sdk.external import NetQASMConnection 
from netqasm.sdk import Qubit, EPRSocket 


STATE_RUNNING_QUANTUM = "RUNNING_QUANTUM"   
STATE_WAITING_PARTITION = "WAITING_PARTITION" 
STATE_DONE = "DONE"


async def handle_quantum_alice(
    ctx: SimpleNamespace,
    s0: np.ndarray,
    s1: np.ndarray,
    ell: int,
    reader: StreamReader,
    writer: StreamWriter,
) -> str:

    ctx.x = np.random.randint(0, 2, size=4*ell).astype(int)
    ctx.theta = np.random.randint(0, 2, size=4*ell).astype(int)

    epr_socket = EPRSocket("Bob")
    with NetQASMConnection("Alice", epr_sockets=[epr_socket], max_qubits=4*ell) as sim_conn:
        for i in range(4*ell):
            #print(f"Alice: teleporting qubit {i}", flush=True)
            epr_half = epr_socket.create_keep(number=1)[0]
            q = Qubit(sim_conn)

            if ctx.x[i] == 1:
                q.X()
            if ctx.theta[i] == 1:
                q.H()

            q.cnot(epr_half)
            q.H()
            m1 = q.measure()
            m2 = epr_half.measure()
            sim_conn.flush()
            m1_val, m2_val = int(m1), int(m2)
            
            writer.write(f"{m1_val}:{m2_val}\n".encode())
            await writer.drain()

    data = await reader.readline()
    msg = data.decode().strip()
    if msg != "MEASURED":
        raise RuntimeError(f"Expected MEASURED, got '{msg}'")

    # now safe to send theta
    theta_str = ",".join(map(str, ctx.theta))
    writer.write(f"{theta_str}\n".encode())
    await writer.drain()
        

    return STATE_WAITING_PARTITION
 
 
async def handle_partition_alice(
    ctx: SimpleNamespace,
    s0: np.ndarray,
    s1: np.ndarray,
    writer: StreamWriter,
    raw_msg: str,
) -> str:


    i0, i1 = raw_msg.split("|", 1)
    I0 = np.array([int(i) for i in i0.split(",") if i], dtype=int)
    I1 = np.array([int(i) for i in i1.split(",") if i], dtype=int)

    t0 = s0 ^ ctx.x[I0]
    t1 = s1 ^ ctx.x[I1]

    t0_str = ",".join(map(str, t0))
    t1_str = ",".join(map(str, t1))

    writer.write(f"{t0_str}|{t1_str}\n".encode())
    await writer.drain()

    return STATE_DONE
 
 
def make_run_alice(s0: np.ndarray, s1: np.ndarray, ell: int):

    async def run_alice(reader: StreamReader, writer: StreamWriter) -> None:
        ctx = SimpleNamespace(x=None, theta=None)
 
        state = STATE_RUNNING_QUANTUM
        while state != STATE_DONE:
 
            if state == STATE_RUNNING_QUANTUM:
                state = await handle_quantum_alice(ctx, s0, s1, ell, reader, writer)
                continue
 
            data = await reader.readline()
            if not data:
                #print(f"Alice [{state}]: connection dropped unexpectedly.")
                break
            raw_msg = data.decode().strip()
            #print(f"Alice [{state}]: received '{raw_msg}'")

            if raw_msg == "ABORT":
                print("Alice: Bob aborted, retrying...")
                state = STATE_RUNNING_QUANTUM
                continue
 
            if state == STATE_WAITING_PARTITION:
                state = await handle_partition_alice(ctx, s0, s1, writer, raw_msg)
 
        #print(f"Alice: OT complete (final state: {state}).")
 
    return run_alice
 

 
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Wrong: python3 alice.py <ell> <s0_bits> <s1_bits>")
        sys.exit(1)
    
    ell      = int(sys.argv[1])
    s0       = np.array([int(b) for b in sys.argv[2]], dtype=int)
    s1       = np.array([int(b) for b in sys.argv[3]], dtype=int)
 
    _here = Path(__file__).parent
    simulaqron_settings.read_from_file(_here / "simulaqron_settings.json")
    network_config.read_from_file(_here / "simulaqron_network.json")
 
    sockets_config = SocketsConfig(network_config, "default", NodeConfigType.APP)
    client = SimulaQronClassicalClient(sockets_config)
 
    #print(f"Alice: starting OT (ell={ell}, s0={s0}, s1={s1})")
    client.run_client("Bob", make_run_alice(s0, s1, ell))




