import multiprocessing
import socket
import sys

from modeling.data import Game
from server.utils import recv_dict, send_dict


def query(port: int) -> bool:
    try:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect(("localhost", port))

        moves = [
            "a2a3",
            "a7a6",
            "b2b3",
            "b7b6",
            "c2c3",
            "c7c6",
            "d2d3",
            "d7d6",
            "e2e3",
            "e7e6",
        ]
        game = Game("180+2", 1234, 2345, None, False, moves, [0] * len(moves))

        send_dict(conn, game.to_dict())
        _ = recv_dict(conn)
        return True
    except:
        return False


def main():
    port = int(sys.argv[1])
    runs = 100

    pool = multiprocessing.Pool(4)
    res = pool.map(query, [port] * runs)

    pool.close()
    pool.join()

    print("SUCCESS:", all(res))


if __name__ == "__main__":
    main()
