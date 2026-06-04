"""
Load and preprocess UNSW-NB15 and CIC-IDS2017 datasets.

TODO — przed uruchomieniem ustaw ścieżki do plików:
  UNSW-NB15:  pobierz CSV z https://research.unsw.edu.au/projects/unsw-nb15-dataset
              i wrzuć do data/raw/unsw/  (oczekiwane pliki: UNSW_NB15_training-set.csv,
              UNSW_NB15_testing-set.csv — albo jeden zbiorczy plik z kolumną podziału)
  CIC-IDS2017: pobierz CSV z https://www.unb.ca/cic/datasets/ids-2017.html
               i wrzuć do data/raw/cic/  (jeden lub więcej plików *.csv)

Jeśli surowe pliki mają inną nazwę kolumny etykiety niż "label" / "Label",
zaktualizuj stałe UNSW_LABEL_COL i CIC_LABEL_COL poniżej.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from config import SEED, PATHS

UNSW_LABEL_COL = "label"       # kolumna 0/1 w UNSW-NB15 (już binarna)
CIC_LABEL_COL = "Label"        # kolumna tekstowa w CIC-IDS2017 ("BENIGN" / atak)
CIC_MAX_ROWS = 200_000         # downsampling CIC jeśli za duży


def load_dataset(name: str, seed: int = SEED):
    """
    Wczytuje i przetwarza zbiór danych.

    Parameters
    ----------
    name : "unsw" | "cic"
    seed : int

    Returns
    -------
    X_train, y_train, X_test, y_test
        X: np.ndarray float32, kształt (n, d), wszystkie wartości w [0, 1]
        y: np.ndarray int,     kształt (n,),   wartości {0, 1}
    """
    cache_path = os.path.join(PATHS["processed"], f"{name}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    if name == "unsw":
        X_train, y_train, X_test, y_test = _load_unsw(seed)
    elif name == "cic":
        X_train, y_train, X_test, y_test = _load_cic(seed)
    else:
        raise ValueError(f"Nieznany zbiór: {name!r}. Użyj 'unsw' lub 'cic'.")

    os.makedirs(PATHS["processed"], exist_ok=True)
    np.savez(cache_path,
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------

def _load_unsw(seed: int):
    raw_dir = PATHS["unsw_raw"]

    train_path = os.path.join(raw_dir, "UNSW_NB15_training-set.csv")
    test_path  = os.path.join(raw_dir, "UNSW_NB15_testing-set.csv")

    _UNSW_DROP = ["id", "attack_cat"]   # id = row index, attack_cat = leakage

    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train = pd.read_csv(train_path).drop(columns=_UNSW_DROP, errors="ignore")
        df_test  = pd.read_csv(test_path).drop(columns=_UNSW_DROP, errors="ignore")
    else:
        # fallback: jeden plik zbiorczy — robimy własny podział 70/30
        csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Brak plików CSV w {raw_dir}. "
                "Pobierz UNSW-NB15 i wrzuć do data/raw/unsw/."
            )
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files],
                       ignore_index=True)
        label_col = _find_label_col(df, UNSW_LABEL_COL)
        y_all = _binarize_unsw(df, label_col)
        df_train, df_test = train_test_split(df, test_size=0.30,
                                             stratify=y_all, random_state=seed)

    label_col = _find_label_col(df_train, UNSW_LABEL_COL)
    X_train, y_train, scaler = _preprocess(df_train, label_col, fit_scaler=True)
    X_test,  y_test,  _      = _preprocess(df_test,  label_col, fit_scaler=False,
                                            scaler=scaler)
    return X_train, y_train, X_test, y_test


def _binarize_unsw(df: pd.DataFrame, label_col: str) -> np.ndarray:
    return df[label_col].astype(int).values


# ---------------------------------------------------------------------------
# CIC-IDS2017
# ---------------------------------------------------------------------------

def _load_cic(seed: int):
    raw_dir = PATHS["cic_raw"]
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Brak plików CSV w {raw_dir}. "
            "Pobierz CIC-IDS2017 i wrzuć do data/raw/cic/."
        )

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files],
                   ignore_index=True)
    df.columns = df.columns.str.strip()

    label_col = _find_label_col(df, CIC_LABEL_COL)

    # binaryzacja: "BENIGN" -> 0, cokolwiek innego -> 1
    df[label_col] = (df[label_col].str.strip().str.upper() != "BENIGN").astype(int)

    # stratyfikowany downsampling jeśli zbyt duży (unikamy groupby-apply z pandas 3)
    if len(df) > CIC_MAX_ROWS:
        _, df = train_test_split(df, test_size=CIC_MAX_ROWS / len(df),
                                 stratify=df[label_col], random_state=seed)
        df = df.reset_index(drop=True)

    df_train, df_test = train_test_split(df, test_size=0.30,
                                         stratify=df[label_col], random_state=seed)

    X_train, y_train, scaler = _preprocess(df_train, label_col, fit_scaler=True)
    X_test,  y_test,  _      = _preprocess(df_test,  label_col, fit_scaler=False,
                                            scaler=scaler)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _find_label_col(df: pd.DataFrame, preferred: str) -> str:
    """Szuka kolumny etykiety (case-insensitive)."""
    cols_lower = {c.strip().lower(): c for c in df.columns}
    if preferred.lower() in cols_lower:
        return cols_lower[preferred.lower()]
    for candidate in ("label", "class", "attack_cat", "category"):
        if candidate in cols_lower:
            return cols_lower[candidate]
    raise KeyError(
        f"Nie znaleziono kolumny etykiety w zbiorze. "
        f"Dostępne kolumny: {list(df.columns)}"
    )


def _preprocess(df: pd.DataFrame, label_col: str,
                fit_scaler: bool, scaler=None):
    """
    Zwraca X (float32, [0,1]), y (int), scaler.
    fit_scaler=True: uczy nowy MinMaxScaler na df i go zwraca.
    fit_scaler=False: stosuje przekazany scaler.
    """
    y = df[label_col].astype(int).values
    df = df.drop(columns=[label_col])

    # usuń kolumny, które nie są numeryczne ani kategoryczne
    for col in df.select_dtypes(include=["object", "str"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.astype(float)

    # NaN -> średnia kolumny, inf -> clip
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(df)

    if fit_scaler:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # clip do [0,1] na wypadek drobnych przekroczeń numerycznych
    X = np.clip(X, 0.0, 1.0).astype(np.float32)
    return X, y, scaler
