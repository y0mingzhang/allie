import json
import sys

from tqdm.auto import tqdm


def main():
    input_file, output_file = sys.argv[1:]
    mean_elos = [0] * 50
    with open(input_file) as f:
        for line in f:
            d = json.loads(line)
            mean_elo = round((int(d["white-elo"]) + int(d["black-elo"])) / 200)
            mean_elos[mean_elo] += 1

    with open(output_file, "w") as fo:
        json.dump(mean_elos, fo, indent=2)


if __name__ == "__main__":
    main()
