#Shared array helpers for the VIMP-for-distribution-shift scripts.
import numpy as np
from model_registry_class import (_as_1d, _as_2d, _rng_seed,
    _as_1d_float, _as_2d_float, _standardize_cols)


def _safe_normalize(v):
    v = np.asarray(v, dtype = float)
    s = np.sum(v)
    return v/s if abs(s) > 1e-12 else v


'''
Infer whether a response is binary, categorical (<= max_levels levels) or continuous.
'''
def infer_response_type(Y, max_levels = 5):
    levels = np.unique(np.asarray(Y).ravel())
    if levels.size == 2:
        return 'binary'
    if levels.size <= max_levels:
        return 'categorical'
    return 'continuous'
