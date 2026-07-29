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








职位描述
1、通过对数据的敏锐洞察以及定性和定量分析，迅速定位内部问题或发现机会；
2、负责适配商业化AI业务数据分析度量体系建设和完善，帮助落地效果的评估，形成 “分析-执行-迭代” 闭环，持续优化业务指标；
3、能根据实际业务完成情况和数据变化，进行较深入的专项数据分析，并形成数据分析报告，给出关键业务建议；
4、挖掘AI产品和业务流量、产品、策略方面的业务机会，驱动商业化AI业务发展。

职位要求
1、统计学、数学、经济学等相关专业，具备扎实的机器学习或数据挖掘理论和技术基础，熟练使用SQL、Python，熟悉常用数据统计和分析方法；
2、熟悉 AI、LLM原理，有一定AI工具相关落地或实操的实践经验并了解模型评估与优化逻辑；具有相关领域较深入的技术应用的经验和能力；
3、良好的沟通能力、团队合作精神、工作规划能力和主动意识；
4、良好的逻辑思维能力、业务解读能力和快速学习能力，能够独立领导完整的数据分析项目；对AI技术工具在业务落地和工作提效中有较大好奇心和探索兴趣。
5、对数字比较敏感，热爱数据分析工作，对数据建模、大模型、AI Agent等相关前沿技术有了解。












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






































