import json
import random
import sys
from os.path import basename, join

from tqdm.auto import tqdm

# subsample elo bins to ~5M games using stats computed on lichess-2022-blitz
sampling_P = {
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 0.567125032241058,
    10: 0.3271509010357401,
    11: 0.2407708558243409,
    12: 0.1836108783129186,
    13: 0.15135926981382145,
    14: 0.12607445059401995,
    15: 0.11212358710264803,
    16: 0.10101730691132735,
    17: 0.09670790246497418,
    18: 0.09324709220598842,
    19: 0.09967881889690315,
    20: 0.11686927045479864,
    21: 0.15700536754250016,
    22: 0.24212928493168928,
    23: 0.4390799027947348,
    24: 0.8289016108708125,
    25: 1.0,
    26: 1.0,
    27: 1.0,
    28: 1.0,
    29: 1.0,
    30: 1.0,
    31: 1.0,
    32: 1.0,
    33: 1.0,
    34: 1.0,
}


def write_line(data, fo):
    json.dump(data, fo)
    fo.write("\n")


def main():
    input_file = sys.argv[1]
    fname = basename(input_file).split(".")[0]
    train_fo = open(f"data/lichess-2022-blitz-sampled/{fname}-train.jsonl", "w")
    val_fo = open(f"data/lichess-2022-blitz-sampled/{fname}-val.jsonl", "w")
    test_fo = open(f"data/lichess-2022-blitz-sampled/{fname}-test.jsonl", "w")

    with open(input_file) as fi:
        for line in tqdm(fi):
            data = json.loads(line)
            elo_sum = int(data["white-elo"]) + int(data["black-elo"])
            mean_elo = round(elo_sum / 200)
            if random.random() >= sampling_P[mean_elo]:
                continue
            unif = random.random()
            if unif < 1e-4:
                write_line(data, val_fo)
            elif unif < 3e-4:
                write_line(data, test_fo)
            else:
                write_line(data, train_fo)

    train_fo.close()
    val_fo.close()
    test_fo.close()

    print("Done!")


if __name__ == "__main__":
    main()
