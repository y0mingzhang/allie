"""One coarse workbench task: collect, validate and analyze the activation screen."""
from . import activation_adaptation, analyze_activation_adaptation
from .service import ROOT
import json


def run(oracle, spec):
    activation_adaptation.test()
    worker = activation_adaptation.run(oracle, spec)
    analyze_activation_adaptation.main()
    result = json.loads((ROOT / 'aug-activation-adaptation-v1/results.json').read_text())
    return dict(worker=worker, fit_cv_selected=result['fit_cv_selected'],
                results={k: v for k, v in result['results'].items()
                         if k in set(result['fit_cv_selected'].values()) | {'parent'}})
