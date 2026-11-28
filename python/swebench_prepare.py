"""Build the SWE-bench Verified tables: instance features (500 x p) and the outcome matrix (500 x submissions).
Outcomes come from the public SWE-bench/experiments leaderboard repo (results.json 'resolved' lists).
Output: real_data/swebench/verified.npz (X, R, repo, sub, instance_id, feature names). Usage: python3 swebench_prepare.py
"""

import json, os, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = os.path.join(os.path.dirname(__file__), "..", "real_data", "swebench")
GH, RAW = (
    "https://api.github.com/repos/SWE-bench/experiments/git/trees/main?recursive=1",
    "https://raw.githubusercontent.com/SWE-bench/experiments/main/",
)


def patch_stats(p):
    lines = p.split("\n")
    return [
        np.log1p(len(re.findall(r"^diff --git", p, re.M))),
        np.log1p(len(re.findall(r"^@@", p, re.M))),
        np.log1p(sum(l.startswith("+") and not l.startswith("+++") for l in lines)),
        np.log1p(sum(l.startswith("-") and not l.startswith("---") for l in lines)),
    ]


def features(d, n_tfidf=600):
    tf = TfidfVectorizer(max_features=n_tfidf, stop_words="english", min_df=5, sublinear_tf=True)
    T = tf.fit_transform(d.problem_statement).toarray()  # unsupervised: instance text only, no outcomes
    num = np.array(
        [
            patch_stats(p)
            + patch_stats(t)[:2]
            + [
                np.log1p(len(s)),
                np.log1p(len(h)),
                len(json.loads(f)),
                np.log1p(len(json.loads(q))),
                (pd.Timestamp(c).year - 2012),
            ]
            for p, t, s, h, f, q, c in zip(
                d.patch, d.test_patch, d.problem_statement, d.hints_text, d.FAIL_TO_PASS, d.PASS_TO_PASS, d.created_at
            )
        ]
    )
    R = pd.get_dummies(d.repo).to_numpy(float)
    X = np.hstack([R, num, T])
    names = (
        list(pd.get_dummies(d.repo).columns)
        + [f"num{i}" for i in range(num.shape[1])]
        + [f"tf:{w}" for w in tf.get_feature_names_out()]
    )
    return (X - X.mean(0)) / (X.std(0) + 1e-9), names


if __name__ == "__main__":
    d = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet")
    tree = json.load(urllib.request.urlopen(GH, timeout=60))["tree"]
    fs = sorted(
        x["path"]
        for x in tree
        if x["path"].startswith("evaluation/verified/") and x["path"].endswith("/results/results.json")
    )
    get = lambda f: (f.split("/")[2], set(json.loads(urllib.request.urlopen(RAW + f, timeout=60).read())["resolved"]))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(get, fs))
    ids = d.instance_id.tolist()
    R = np.array([[i in r for i in ids] for _, r in res], bool)
    X, names = features(d)
    keep = R.sum(1) > 0
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT, "verified.npz"),
        X=X.astype(np.float32),
        R=R[keep],
        repo=d.repo.to_numpy(str),
        sub=np.array([s for (s, _), k in zip(res, keep) if k]),
        instance_id=np.array(ids),
        names=np.array(names),
    )
    print(
        "instances",
        len(ids),
        "features",
        X.shape[1],
        "submissions kept",
        keep.sum(),
        "of",
        len(keep),
        "resolve rate first/last",
        R[keep][0].mean(),
        R[keep][-1].mean(),
    )
