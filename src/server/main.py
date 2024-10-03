import socket
import sys
from os.path import join

import chess
import torch
from omegaconf import OmegaConf

from evaluation.decode import DECODE_STRATEGIES, Maia
from modeling.data import Game, UCITokenizer
from modeling.model import initialize_model
from server.test import algorithm_test
from server.utils import DEFAULT_SYSTEM, SYSTEMS, recv_dict, send_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = "pretrain_config/medium.yaml"


@torch.inference_mode(True)
def main():
    assert len(sys.argv) == 2, "provide a system alias"
    alias = sys.argv[1]
    assert alias in SYSTEMS, f"unknown alias {alias}"
    system = SYSTEMS[alias]
    config = OmegaConf.load(CONFIG)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = UCITokenizer(**config.data_config.tokenizer_config)

    model = initialize_model(tokenizer, **config.model_config)
    model = model.to(DEVICE)
    model.eval()
    ckpt_path = join(config.trainer_config.output_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    if system.algorithm == "maia":
        default_algorithm = DECODE_STRATEGIES[DEFAULT_SYSTEM.algorithm](
            model, tokenizer, **DEFAULT_SYSTEM.algorithm_config
        )
        algorithm = Maia(opening_strategy=default_algorithm, **system.algorithm_config)
    else:
        algorithm = DECODE_STRATEGIES[system.algorithm](
            model, tokenizer, **system.algorithm_config
        )

    print("system initialized", system)
    algorithm_test(algorithm)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", system.port))
    print("starts listening on port", system.port)
    sock.listen(64)
    while True:
        try:
            conn, addr = sock.accept()
            print("accepted", addr)
            try:
                data = recv_dict(conn)
            except RuntimeError:
                print("socket connection broken, listen to new connections")
                continue
            print("received", data)
            game = Game(**data)
            if system.elo is None:
                # set elo to opponent elo
                system_elo = (
                    game.black_elo if game.board.turn == chess.WHITE else game.white_elo
                )
            else:
                system_elo = system.elo

            assert system_elo is not None

            if game.board.turn == chess.WHITE:
                game.white_elo = system_elo
            else:
                game.black_elo = system_elo
            print("decoding game", game.to_dict())
            result = algorithm.decode_full(game)
            move_result = {
                "move": result.move,
                "think_time": result.think_time,
                "resigned": result.resigned,
            }
            send_dict(conn, move_result)
            print("sent", move_result)
            conn.close()
            print("closed", addr)

        except KeyboardInterrupt:
            algorithm.cleanup()
            print("interrupted and exiting..")
            break

        except Exception as e:
            algorithm.cleanup()
            raise e


if __name__ == "__main__":
    main()
