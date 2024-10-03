from os.path import join
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm.auto import tqdm


def download_pgn(gameid):
    response = requests.get(f"https://lichess.org/game/export/{gameid}?literate=1")
    assert response.status_code == 200
    with open(join("data", "lichess-puzzles", "pgn", f"{gameid}.pgn"), "wb") as f:
        f.write(response.content)


def main():
    df = pd.read_csv("data/lichess-puzzles/lichess_db_puzzle.csv")
    sample_size = 100
    with tqdm(total=sample_size * 5) as pbar:
        for x in range(1, 6):
            subset = df[df["Themes"].map(lambda s: f"mateIn{x}" in s)].sample(
                sample_size
            )
            subset.to_csv(f"data/lichess-puzzles/mateIn{x}.csv")
            for game_url in subset["GameUrl"]:
                gameid = urlparse(game_url).path.split("/")[1]
                download_pgn(gameid)
                pbar.update(1)


if __name__ == "__main__":
    main()
