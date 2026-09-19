"""Training-equivalent CM using a frozen golden compute-optimal learning law.

Anchor one curve to this checkpoint's raw CE at C0, then invert each method's CE.
Report C_eq(method)/C0 and C_eq(method)/C_eq(cheap) using that same curve.
This estimates training-FLOP equivalence, not net training+serving savings.
"""
import hashlib
import json
import math
from pathlib import Path

BASE=Path('/data/group_data/dei-group/yimingz3/allie')
OUT=Path(__file__).resolve().parents[1]/'results/search-v1/training-cm-laws.json'
METRICS={'macro':'strat_macro','expert_macro':'strat_expert'}


def excess(law,c):
    """Optimum of A*(N/1e7)^-alpha+B*(D/1e8)^-beta with N*D=C."""
    a,b=law['alpha'],law['beta'];scaled=c/1e15
    x=math.exp((math.log(a*law['A']/(b*law['B']))+b*math.log(scaled))/(a+b))
    return law['A']*x**(-a)+law['B']*(scaled/x)**(-b)


def multiplier(law,c,raw_loss,method_loss):
    residual=excess(law,c)
    target=residual+(method_loss-raw_loss)
    if target<=0:return None
    gamma=law['alpha']*law['beta']/(law['alpha']+law['beta'])
    return math.exp(math.log(residual/target)/gamma)


def annotate(methods,raw_name='official_raw',cheap_name='legal'):
    frozen=json.loads(OUT.read_text());result={}
    for name,metrics in methods.items():
        row=dict(metrics)
        for metric in METRICS:
            law=frozen['metrics'][metric]['law'];c=frozen['budget_nd']
            raw=methods[raw_name][metric]
            total=multiplier(law,c,raw,metrics[metric])
            cheap=multiplier(law,c,raw,methods[cheap_name][metric])
            row[metric+'_training_eq_cm']=total
            row[metric+'_training_eq_cm_vs_cheap']=None if total is None or cheap is None else total/cheap
            row[metric+'_equivalent_nd']=None if total is None else c*total
        result[name]=row
    return result


def snapshot():
    record=dict(budget_nd=3e16,training_flops_convention='ND; conventional6ND cancels in the ratio',
        allocation='Compute-optimal N and D; not fixed-N continued training',
        anchor='One vertical shift to this checkpoint official raw CE at C0; same curve for all methods',
        interpretation='Estimated training-equivalent CM; report inference cost separately',
        caveats=['Local law shape transferred from isoflop-v1 to this data/aux checkpoint',
                 'Evaluation bootstrap does not include scaling-fit uncertainty',
                 'Do not apply these golden curves to dev/dev_expert losses'],metrics={})
    for metric,key in METRICS.items():
        path=BASE/f'results/recipe10x/strat-eval-v1/fit-{key}.json';fit=json.loads(path.read_text())
        record['metrics'][metric]=dict(law=fit['ours_law'],alternative_shared_floor=fit['shared_E']['fit']['ours'],
            path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if OUT.exists():assert json.loads(OUT.read_text())==record,'Never silently change CM laws'
    else:OUT.write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    from scipy.optimize import minimize_scalar
    fit=snapshot()
    for metric,entry in fit['metrics'].items():
        law=entry['law'];c=fit['budget_nd'];a,b=law['alpha'],law['beta']
        def numeric(c):
            return minimize_scalar(lambda ln:law['A']*(math.exp(ln)/1e7)**(-a)+
                law['B']*(c/math.exp(ln)/1e8)**(-b),bounds=(math.log(1e4),math.log(c/1e4)),method='bounded').fun
        assert abs(excess(law,c)-numeric(c))<1e-10
        assert multiplier(law,c,1.5,1.5)==1.
        for delta in (-.02,.005,.01,.02):
            cm=multiplier(law,c,1.5,1.5-delta)
            assert abs(numeric(cm*c)-(numeric(c)-delta))<1e-10
        print(metric,'illustrative 0.01-nat golden gain ->',multiplier(law,c,1.5,1.49),'training-equivalent CM; not measured search performance')
    print('PASS: analytic optimum and inverse agree with numerical minimization')
