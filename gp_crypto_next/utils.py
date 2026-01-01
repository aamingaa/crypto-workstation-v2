"""Utilities that are required by gplearn.

Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers

import numpy as np
from joblib import cpu_count
import pandas as pd

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

    # # 改为支持int64的版本，之前的版本仅仅支持int32
    # if seed is None:
    #     return np.random.default_rng()
    # if isinstance(seed, (numbers.Integral, np.integer)):
    #     return np.random.default_rng(seed)
    # if isinstance(seed, (np.random.RandomState, np.random.Generator)):
    #     return seed
    # raise ValueError('%r cannot be used to seed a numpy.random.Generator instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

def norm(x: np.ndarray, rolling_zscore_window=2000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # mean = pd.Series(x).rolling(2000, min_periods=1).mean().values
    std = pd.Series(x).rolling(rolling_zscore_window, min_periods=1).std().values
    # x_value = (x - mean) / np.clip(np.nan_to_num(std),
    #                                a_min=1e-6, a_max=None)
    x_value = (x ) / np.clip(np.nan_to_num(std),
                                    a_min=1e-6, a_max=None)
    # x_value = np.clip(x_value, -6, 6)
    x_value = np.nan_to_num(x_value, nan=0.0, posinf=0.0, neginf=0.0)
    return x_value

# ---- 辅助函数（仅本作用域内使用）----
def _safe_div(numer: np.ndarray, denom: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """数值安全除法，避免 0 除与极端值。"""
    return np.asarray(numer, dtype=np.float64) / np.maximum(np.asarray(denom, dtype=np.float64), eps)

