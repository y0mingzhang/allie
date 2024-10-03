import json
import socket
import time
from dataclasses import dataclass


def send_dict(conn: socket.socket, dict: dict) -> None:
    conn.sendall(json.dumps(dict).encode("utf-8"))


def recv_dict(conn: socket.socket) -> dict:
    chunks = []
    while True:
        try:
            chunk = conn.recv(8192)
            if chunk == b"":
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            msg = b"".join(chunks)
            data = json.loads(msg.decode("utf-8"))
            return data
        except json.JSONDecodeError:
            continue  # incomplete message


@dataclass(frozen=True)
class System:
    port: int
    elo: int | None
    algorithm: str
    algorithm_config: dict


MAIA = System(20602, None, "maia", {})
DEFAULT_SYSTEM = ALLIE_POLICY = System(20600, None, "policy", {"sample": True})
ALLIE_STRONG = System(20601, 2500, "policy", {"sample": False})
ALLIE_ADAPTIVE_SEARCH = System(
    20607,
    None,
    "adaptive-mcts",
    {"mean_n_sims": 50, "c_puct": 1.25, "sample": "rating-based"},
)
ALLIE_SEARCH = System(
    20608, None, "mcts", {"n_sims": 50, "c_puct": 1.25, "sample": "rating-based"}
)

SYSTEMS = {
    "maia": MAIA,
    "humanlike": ALLIE_POLICY,
    "grandmaster": ALLIE_STRONG,
    "humanlike-adaptive-rb": ALLIE_ADAPTIVE_SEARCH,
    "humanlike-mcts-rb": ALLIE_SEARCH,
}

if __name__ == "__main__":
    print(" ".join(SYSTEMS))
