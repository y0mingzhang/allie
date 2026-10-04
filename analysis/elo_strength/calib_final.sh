#!/bin/bash
# calib_final.sh CALIB_DIR SEARCH_DIR TAG: the T = 1 search rule's fits (searching in every time control,
# and with no search in bullet), the cross-entropy table and the figure on one model's search outputs
set -uo pipefail
C=$1 S=$2 T=$3
py() { /home/yimingz3/src/allie/.venv/bin/python "$@"; }
common="--dists $S --families search --hw 40 --ce-weight 10 --ladder 0,32,128,256"
py calib_rule.py $C $common --out rule-$T-search.json > $C/rule-$T-search.log 2>&1 &
py calib_rule.py $C $common --no-bullet-search --out rule-$T-nobullet.json > $C/rule-$T-nobullet.log 2>&1 &
wait
grep -hE "^current|^search|all:|strength-only" $C/rule-$T-search.log $C/rule-$T-nobullet.log
py - $C $T <<'PY'
import json, sys
c, t = sys.argv[1:]
rules = {}
for name, label, colour, dash, nb in (("search", "Search, every time control", "#0d9488", "--", False),
                                      ("nobullet", "Search, none in bullet", "#2563eb", "-", True)):
    r = json.load(open(f"{c}/rule-{t}-{name}.json"))["search"]["all"]["params"]
    rules[name] = dict(params=r, hw=40, ladder=[0, 32, 128, 256], no_bullet=nb, label=label, colour=colour, dash=dash)
json.dump(rules, open(f"{c}/rules-{t}.json", "w"), indent=1)
print(json.dumps({k: v["params"] for k, v in rules.items()}))
PY
py calib_ce.py $C --dists $S --rules $C/rules-$T.json --out ce-$T.json
py calib_fit.py $C --positions positions-human.npz --mpv mpv-human --dists $S --rules rules-$T.json --figure --out calib-$T.json > $C/fit-$T.log 2>&1
cp $C/candidates.png $C/candidates-$T.png; cp $C/candidates.svg $C/candidates-$T.svg
grep -A12 "Calibration error" $C/fit-$T.log | head -16
echo done
