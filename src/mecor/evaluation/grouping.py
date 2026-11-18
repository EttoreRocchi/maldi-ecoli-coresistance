"""Patient-aware folds and subsamples.

Without patient groups every function reproduces exactly the splits used so far
(RepeatedStratifiedKFold, StratifiedKFold, train_test_split with the same seeds), so existing results stay
reproducible. With patient groups (DRIAMS-A, from the patient_no column of id/<year>/<year>_strat.csv) no
patient is ever split between training and test, in the outer folds, in the inner folds and in subsamples.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split

Split = Tuple[np.ndarray, np.ndarray]


def profile_strata(Y: pd.DataFrame) -> np.ndarray:
    """Joint resistance profile of each isolate (for example '010'), used for stratification."""
    return Y.apply(lambda r: "".join(r.astype(str).values), axis=1).values


def load_driams_ids(config, index) -> pd.DataFrame:
    """patient_no and case_no of each isolate (aligned with index), from id/<year>/<year>_strat.csv."""
    base = Path(config.metadata_path)
    if base.is_file():
        base = base.parent.parent
    frames = [pd.read_csv(base / y / f"{y}_strat.csv", usecols=["code", "patient_no", "case_no"], low_memory=False)
              for y in config.driams_years]
    ids = pd.concat(frames).drop_duplicates("code").set_index("code")
    ids.index = ids.index.astype(str)
    ids = ids.reindex([str(c) for c in index])
    if ids["patient_no"].isna().any():
        missing = list(ids.index[ids["patient_no"].isna()][:5])
        raise ValueError(f"{int(ids['patient_no'].isna().sum())} isolates without patient_no, for example {missing}")
    return ids


def patient_case_components(patients, cases) -> np.ndarray:
    """Group label of each isolate: connected components of patients and cases, so that neither a patient nor a case
    (a hospital case, the unit of maldi-learn's case_based_stratification) is ever split. A missing case_no links nothing."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, c in zip(patients, cases):
        rp = find(("p", str(p)))
        if not pd.isna(c):
            rc = find(("c", str(c)))
            if rc != rp:
                parent[rc] = rp
    roots = [find(("p", str(p))) for p in patients]
    labels = {r: f"G{i:05d}" for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([labels[r] for r in roots])


def load_patient_groups(config, index) -> Optional[np.ndarray]:
    """Group of each isolate (aligned with index): patients joined through shared cases, so that no patient and no case
    is split between training and test. None when the config does not group by patient (every site but driams_a_grouped)."""
    if not getattr(config, "group_by_patient", False):
        return None
    ids = load_driams_ids(config, index)
    return patient_case_components(ids["patient_no"].values, ids["case_no"].values)


def outer_splits(strata, groups=None, n_splits: int = 5, n_repeats: int = 3, seed: int = 42) -> List[Split]:
    """Outer folds in the order repeat 0 folds 0..k-1, repeat 1, ..."""
    x0 = np.zeros(len(strata))
    if groups is None:
        return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(x0, strata))
    out: List[Split] = []
    for r in range(n_repeats):
        out += list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r).split(x0, strata, groups))
    return out


def inner_splits(strata, groups=None, n_splits: int = 3, seed: int = 42) -> Optional[List[Split]]:
    """Patient-grouped inner folds (indices relative to the training rows), or None to keep the default inner CV."""
    if groups is None:
        return None
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(strata)), strata, groups))


def subsample(n_target: int, strata, groups=None, seed: int = 0) -> np.ndarray:
    """Training subsample of about n_target isolates.

    Without groups: the stratified train_test_split used so far. With groups: whole patients, stratified by the
    patient's resistance profile (a drug counts as resistant if any isolate of the patient is), drawn in a random
    order that interleaves the strata in proportion to their size, until n_target isolates are reached. A patient
    is never split, the prevalences follow the cohort's at every size, and the size can exceed n_target by the
    isolates of the last patient drawn.
    """
    n = len(strata)
    if groups is None:
        return train_test_split(np.arange(n), train_size=n_target, stratify=strata, random_state=seed)[0]
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": groups, "s": np.asarray(strata).astype(str)})
    profile = df.groupby("g")["s"].agg(lambda v: "".join(max(c) for c in zip(*v)))   # '1' (resistant) wins per drug
    sizes = df.groupby("g").size()
    key = {}
    for _, members in profile.groupby(profile):   # strata in sorted order, so the draw depends on the seed only
        ids = members.index.to_numpy().copy(); rng.shuffle(ids); u = rng.random()
        for j, p in enumerate(ids):
            key[p] = (j + u) / len(ids)   # stratified systematic positions in [0, 1)
    chosen, total = [], 0
    for p in sorted(key, key=key.get):
        if total >= n_target:
            break
        chosen.append(p)
        total += int(sizes[p])
    return np.flatnonzero(np.isin(groups, chosen))


def check_no_patient_overlap(splits: List[Split], groups) -> None:
    """Raise if any patient appears on both sides of a split."""
    if groups is None:
        return
    groups = np.asarray(groups)
    for k, (tr, te) in enumerate(splits):
        shared = set(groups[tr]) & set(groups[te])
        if shared:
            raise AssertionError(f"split {k}: {len(shared)} patients in both training and test")
