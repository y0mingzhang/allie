"""Collect and analyze the correlated-history screen as one resident task."""
import json
from .service import ROOT
from . import activation_ridge_v2 as collect, analyze_activation_ridge as analyze

def run(oracle,spec):
    collect.test();collect.test_ridge()
    collect.run(oracle,spec);analyze.main()
    d=json.loads((ROOT/'aug-activation-ridge-v1/results.json').read_text())
    return {k:d[k] for k in ('fit_cv_selected','analysis_seconds')}
