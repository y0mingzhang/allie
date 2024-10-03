import sys

import chess
import pandas as pd
from tqdm.auto import tqdm

from evaluation.utils import compute_centipawn_loss_and_accuracy, get_stockfish_engine


def annotate(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    hcl = []
    hla = []
    mic = []
    miep = []
    mip = []
    mit = []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), file=sys.stdout):
        board = chess.Board()
        human_centipawn_loss = []
        human_lichess_accuracy = []
        move_is_castling = []
        move_is_en_passant = []
        move_is_promotion = []
        for move_uci in row["moves-uci"].split():
            loss, acc = compute_centipawn_loss_and_accuracy(board, move_uci)
            human_centipawn_loss.append(loss)
            human_lichess_accuracy.append(acc)

            move = chess.Move.from_uci(move_uci)
            move_is_castling.append(board.is_castling(move))
            move_is_en_passant.append(board.is_en_passant(move))
            move_is_promotion.append(move.promotion is not None)
            board.push_uci(move_uci)

        move_is_threefold = [False] * len(board.move_stack)
        if board.can_claim_threefold_repetition():
            move_is_threefold[-1] = True

        hcl.append(human_centipawn_loss)
        hla.append(human_lichess_accuracy)
        mic.append(move_is_castling)
        miep.append(move_is_en_passant)
        mip.append(move_is_promotion)
        mit.append(move_is_threefold)

    get_stockfish_engine().quit()
    dataframe = dataframe.copy()
    dataframe["human-centipawn-loss"] = hcl
    dataframe["human-lichess-accuracy"] = hla
    dataframe["move-is-castling"] = mic
    dataframe["move-is-en-passant"] = miep
    dataframe["move-is-promotion"] = mip
    dataframe["move-is-threefold"] = mit

    return dataframe


def main():
    assert len(sys.argv) == 2
    dataframe = pd.read_json(sys.argv[1], lines=True)
    annotated = annotate(dataframe)
    assert sys.argv[1].endswith(".jsonl")
    annotated.to_json(
        sys.argv[1][:-6] + "-annotated.jsonl", lines=True, orient="records"
    )


if __name__ == "__main__":
    main()
