#!/bin/bash

set -Eeuo pipefail

data_dir=/data/datasets/models/hf_cache/tmp-lichess
mkdir -p $data_dir/lichess

year=2022
for month in $(seq 1 12)
do
    month_format=`printf %02d $month`
    URL=https://database.lichess.org/standard/lichess_db_standard_rated_${year}-${month_format}.pgn.zst
    echo working on $URL
    curl $URL -s -o $data_dir/lichess/${year}-${month_format}.pgn.zst -C - &
done

wait
echo "Done!"
