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
    """
    STATE_RUNNING_QUANTUM handler.
 
    Steps:
    1. Sample x uniformly from {0,1}^{2ℓ} and θ uniformly from {0,1}^{2ℓ}.
       Store both in ctx.
 
    2. Open EPRSocket("Bob") and NetQASMConnection("Alice", ...).
       For each qubit j in 0..2ℓ-1:
         a. Receive one EPR half via epr_socket.create_keep(number=1)[0]
            (one at a time to keep things simple — see teleportation example)
         b. Prepare the BB84 state on a fresh Qubit(conn):
              x[j]=0, θ[j]=0 → |0⟩  (do nothing)
              x[j]=1, θ[j]=0 → |1⟩  (apply X)
              x[j]=0, θ[j]=1 → |+⟩  (apply H)
              x[j]=1, θ[j]=1 → |−⟩  (apply X then H)
         c. Run teleportation circuit on (bb84_qubit, epr_half):
              bb84_qubit.cnot(epr_half)
              bb84_qubit.H()
              m1 = bb84_qubit.measure()
              m2 = epr_half.measure()
         d. conn.flush()
         e. Send correction bits as "m1:m2\n" to Bob over classical channel
            Bob applies corrections and now holds the BB84 state.
 
    3. Wait for Bob's "MEASURED\n" acknowledgement on reader.
       This is the critical security step — do NOT send θ before this.
 
    4. Send θ as comma-separated string + newline over writer.
       Example: "0,1,0,1,1,0\n"
 
    5. Store ctx.x = x, ctx.theta = θ.
 
    Returns: STATE_WAITING_PARTITION
    """

    ctx.x = np.random.randint(0, 2, size=2*ell).astype(int)
    ctx.theta = np.random.randint(0, 2, size=2*ell).astype(int)

    epr_socket = EPRSocket("Bob")
    with NetQASMConnection("Alice", epr_sockets=[epr_socket], max_qubits=2*ell) as sim_conn:
        for i in range(2*ell):

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
        

    #transition to the next state
    return STATE_WAITING_PARTITION


 
 
async def handle_partition_alice(
    ctx: SimpleNamespace,
    s0: np.ndarray,
    s1: np.ndarray,
    ell: int,
    writer: StreamWriter,
    raw_msg: str,
) -> str:
    """
    STATE_WAITING_PARTITION handler.
 
    raw_msg contains Bob's partition "(I0)|(I1)" as comma-separated indices.
    Example: "0,3,5|1,2,4,6,7"
    Empty set encoded as empty string on that side: "|1,2,3,4"
 
    Steps:
    1. Parse I0 and I1 from raw_msg into np.ndarray of int.
    2. Compute:
         t0 = s0 XOR ctx.x[I0]   (element-wise XOR, both are bit arrays)
         t1 = s1 XOR ctx.x[I1]
    3. Send t0 and t1 as "t0_bits|t1_bits\n" where each side is
       comma-separated bits.
       Example: "0,1,1,0|1,0,0,1\n"
 
    Returns: STATE_DONE
    """
    # TODO: implement
    pass
 
 
 
def make_run_alice(s0: np.ndarray, s1: np.ndarray, ell: int):
    """
    Returns Alice's event-loop coroutine for one OT execution.
    s0, s1: ell-bit arrays (Alice's two input strings)
    ell: bit length of each string
    """
 
    async def run_alice(reader: StreamReader, writer: StreamWriter) -> None:
        ctx = SimpleNamespace(x=None, theta=None)
 
        state = STATE_RUNNING_QUANTUM
        while state != STATE_DONE:
 
            if state == STATE_RUNNING_QUANTUM:
                state = await handle_quantum_alice(ctx, s0, s1, ell, reader, writer)
                continue
 
            data = await reader.readline()
            if not data:
                print(f"Alice [{state}]: connection dropped unexpectedly.")
                break
            raw_msg = data.decode().strip()
            print(f"Alice [{state}]: received '{raw_msg}'")
 
            if state == STATE_WAITING_PARTITION:
                state = await handle_partition_alice(ctx, s0, s1, ell, writer, raw_msg)
 
        print(f"Alice: OT complete (final state: {state}).")
 
    return run_alice
 

 
if __name__ == "__main__":
    """
    Test OT in isolation.
    Usage: python3 alice.py <ell> <s0_bits> <s1_bits>
    Example: python3 alice.py 4 1011 0100
    """
    if len(sys.argv) != 4:
        print("Usage: python3 alice.py <ell> <s0_bits> <s1_bits>")
        sys.exit(1)
 
    ell      = int(sys.argv[1])
    s0       = np.array([int(b) for b in sys.argv[2]], dtype=int)
    s1       = np.array([int(b) for b in sys.argv[3]], dtype=int)
 
    _here = Path(__file__).parent
    simulaqron_settings.read_from_file(_here / "simulaqron_settings.json")
    network_config.read_from_file(_here / "simulaqron_network.json")
 
    sockets_config = SocketsConfig(network_config, "default", NodeConfigType.APP)
    client = SimulaQronClassicalClient(sockets_config)
 
    print(f"Alice: starting OT (ell={ell}, s0={s0}, s1={s1})")
    client.run_client("Bob", make_run_alice(s0, s1, ell))




