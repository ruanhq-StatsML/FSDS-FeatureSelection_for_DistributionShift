#Chronoberg Synthetic DID or something like this:
import os
os.chdir(
    "/Users/heqiaoruan/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
)
from benchmark_config import MODEL_REGISTRY
import numpy as np
import pandas as pd
from benchmark_methods import *
from itertools import product
from sklearn.decomposition import PCA
from grf_vimp_causalForest import cf_variable_importance
from VIMP_drperm_benchmark import DRPerm_LOCO
from LOCO_vimp_r_risk import vimp_loco_r_risk
from real_data_utils import *
from adversarial_perturbation_distribution_shift_whole import impose_adv_shift
import azcausal
import causalpy as cp
import SyntheticControlMethods
import statspai 
import azcausal 
import econml
import mcf
import datashifts


#loading the chronoberg dataset and then localize the distribution shift across:

EMB_DIR = 'real_data/yearly_extracted_batch'
EMB_NAME = 'sentence_chronoberg_processed_Emb_gemma_embedding2.npy'
START_YEAR = np.round(np.linspace(1750, 1970, 23)).astype(int)


SCENARIO = "chronoberg"
SUBSAMPLE = 2000
PCA_N = 30
RANDOM_STATE = 2000




#with 10 years in between:
for pair_idx, start_year in enumerate(START_YEAR):
    end_year = int(start_year) + 10
    dataset_id = f"{start_year}_{end_year}"
    #start year path:
    path1 = os.path.join(EMB_DIR, f"year_{int(start_year)}_{EMB_NAME}")
    path2 = os.path.join(EMB_DIR, f"year_{int(end_year)}_{EMB_NAME}")
    df1_arr = np.load(path1, allow_pickle = True)
    df2_arr = np.load(path2, allow_pickle = True)
    df1_X = arr1[:, :-1]
    df2_X = arr2[:, :-1]
    df1_Y = arr1[:, -1].ravel()
    df2_Y = arr2[:, -1].ravel()
    df1, df2 = pca_pair(df1_arr, df2_arr, n_components = 30,
        random_state = 2026, subsample = 200)
    #Then performing the localization here:





#conduct the mcf here:
import mcf
import econml 
import CausalML
import numpy as np
import pandas as pd
import zipfile
import io
from urllib import request
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


















P(A1) = 1/6 -> P(B2) = 5/6 * 1/5 -> B 
P(A3) = (4/6) * (1/4) -> the marginal probability is 1/6 here
the marginal probability for each of the bulletin here.

def russian_roulette(strategy, n_sim = 100000):
    survival = 0.0
    for _ in range(n_sim):
    	#random sample from 1 to 6:
    	bullet = random.randint(1, 6)
    	if strategy == 'first':
    	    deaths = [1, 3, 5]
    	else:
    	    deaths = [2, 4, 6]
        if strategy == 'first':
            deaths = [1, 3, 5]
        else:
            deaths = [2, 4, 6]
        if bullet not in deaths:
            survival += 1
    return survival/n_sim
    return survival/n_sim






































