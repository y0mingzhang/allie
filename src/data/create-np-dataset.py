import json
import random
import sys

import numpy as np
from tqdm.auto import tqdm

from modeling.data import UCITokenizer, json_to_game


def main():
    input_file, output_file = sys.argv[1:]
    tokenizer = UCITokenizer()

    with open(input_file) as fi:
        all_tokens = []
        for line in tqdm(fi):
            game = json_to_game(json.loads(line))
            # 5% of the games don't contain ELO
            add_elo = random.random() > 0.05
            # 5% of the games has masked out time control token
            add_time_control = random.random() > 0.05
            tokens_np = tokenizer.tokenize(game, add_elo, add_time_control)
            all_tokens.append(tokens_np)

        random.shuffle(all_tokens)
        all_tokens_np = np.concatenate(all_tokens)
        mmap = np.memmap(
            output_file, dtype=np.uint32, mode="w+", shape=all_tokens_np.shape
        )
        mmap[:] = all_tokens_np
        mmap.flush()

    print("Done!")


if __name__ == "__main__":
    main()
