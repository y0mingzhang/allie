"""Visible registry of every method scored on the shared golden sample."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    rows=[]
    for folder in sorted(ROOT.glob('golden-*')):
        path=folder/'results.json'
        if not path.exists():continue
        data=json.loads(path.read_text());methods=data.get('methods',{})
        if not methods:continue
        plan=folder/'plan.json'
        for method in methods:
            rows.append(dict(study=folder.name,method=method,
                report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else None))
    (ROOT/'golden-methods.json').write_text(json.dumps(dict(methods=rows,note='Includes duplicate controls; do not interpret repeated rows as independent evidence.'),indent=2)+'\n')
    lines=['# Golden method registry','','Every method in a completed golden report, including duplicate controls. The sample has been reused; per-study preregistration does not eliminate cumulative adaptive benchmark bias. A frozen final winner needs fresh confirmation.','','| Study | Method |','|---|---|']
    lines += [f"| {r['study']} | {r['method']} |" for r in rows]
    (ROOT/'GOLDEN_METHODS.md').write_text('\n'.join(lines)+'\n')
    print('Registered',len(rows),'method/control entries')


if __name__=='__main__':main()
