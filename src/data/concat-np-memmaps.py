import sys

import numpy as np
from tqdm.auto import tqdm


def main():
    mmaps = [np.memmap(f, dtype=np.uint32) for f in sys.argv[1:-1]]
    total_len = sum(map(len, mmaps))

    out_mmap = np.memmap(sys.argv[-1], dtype=np.uint32, mode="w+", shape=(total_len,))

    offset = 0
    for mmap in tqdm(mmaps):
        out_mmap[offset : offset + len(mmap)] = mmap
        offset += len(mmap)

    out_mmap.flush()
    print("Done!")


if __name__ == "__main__":
    main()
