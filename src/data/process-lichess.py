import argparse
import io
import json
import os
import time
from multiprocessing import Process, Queue, Value
from os.path import dirname, join

import pandas as pd
import zstandard as zstd
from chess import pgn
from tqdm.auto import tqdm


def moves_to_san(game):
    exporter = pgn.StringExporter(
        columns=None, headers=False, variations=False, comments=False
    )
    return game.accept(exporter)


def parse_time_control(tc: str) -> list[int, int]:
    try:
        total_time, increment = map(int, tc.split("+"))
        return [total_time, increment]
    except Exception as e:
        raise Exception(f"Failed to parse time control {tc}")


def extract_moves_and_time(game):
    moves = []
    move_seconds = []

    total_time, increment = parse_time_control(game.headers["TimeControl"])
    time_rem = [total_time, total_time]
    turn = 0

    node = game.next()
    while node:
        clock = int(node.clock())
        time_usage, time_rem[turn] = time_rem[turn] - clock + increment, clock
        moves.append(node.move.uci())
        move_seconds.append(time_usage)
        node = node.next()
        turn = 1 - turn

    return " ".join(moves), move_seconds


def game_to_dict(game):
    moves_uci, move_seconds = extract_moves_and_time(game)
    return {
        "game-id": game.headers["Site"],
        "moves-uci": moves_uci,
        "moves-seconds": move_seconds,
        "event": game.headers["Event"],
        "result": game.headers["Result"],
        "white-elo": game.headers["WhiteElo"],
        "black-elo": game.headers["BlackElo"],
        "termination": game.headers["Termination"],
        "time-control": game.headers["TimeControl"],
        "opening": game.headers["Opening"],
    }


def parse_games(in_queue, out_queue, shutdown_signal):
    while True:
        try:
            game_string = in_queue.get_nowait()
            game = pgn.read_game(io.StringIO(game_string))
            game_dict = game_to_dict(game)
            if (
                "Blitz" in game_dict["event"]
                and len(game_dict["moves-uci"].split()) > 8
            ):
                out_queue.put(json.dumps(game_dict) + "\n")
        except:
            if in_queue.empty() and shutdown_signal.value:
                return
            time.sleep(0.0001)


def write_games(output_file, out_queue, shutdown_signal):
    with open(output_file, "w") as f:
        while True:
            try:
                json_string = out_queue.get_nowait()
                f.write(json_string)
            except:
                if out_queue.empty() and shutdown_signal.value:
                    return
                time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--n_procs", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(dirname(args.output_file), exist_ok=True)
    in_queue = Queue(maxsize=1024)
    out_queue = Queue(maxsize=1024)
    parser_shutdown = Value("d", 0)
    writer_shutdown = Value("d", 0)

    parser_procs = []
    for _ in range(args.n_procs):
        proc = Process(target=parse_games, args=(in_queue, out_queue, parser_shutdown))
        proc.start()
        parser_procs.append(proc)

    writer_proc = Process(
        target=write_games, args=(args.output_file, out_queue, writer_shutdown)
    )
    writer_proc.start()

    dctx = zstd.ZstdDecompressor()

    with tqdm() as pbar:
        with open(args.dump_file, "rb") as fb:
            reader = dctx.stream_reader(fb)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            lines = []
            for line in text_stream:
                if line.startswith("[Event") and lines:
                    in_queue.put("".join(lines))
                    lines = []
                    pbar.update(1)
                lines.append(line)

            in_queue.put("".join(lines))
            pbar.update(1)

    parser_shutdown.value = 1
    for proc in parser_procs:
        proc.join()

    writer_shutdown.value = 1
    writer_proc.join()

    print("Done!")


if __name__ == "__main__":
    main()
