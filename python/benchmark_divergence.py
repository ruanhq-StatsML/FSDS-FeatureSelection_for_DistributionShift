#benchmark the fisher's divergence method:
import numpy as np
from copy import copy
from sklearn.utils import check_array, check_random_state
from scipy.stats import ks_2samp as ks_stat




class FisherDivergence:
    def __init__(self, density_mdoel, n_expectation = 100):
        self.density_model = density_model
        self.n_expectation = n_expectation
        self.p_hat_ = None
        self.q_hat_ = None
    def fit(self, X, Y):
        X



@staticmethod
def _calculate_1d_gaussian_conditional(x, feature_idx, joint_mean, joint_cov,
    random_state = None):
    