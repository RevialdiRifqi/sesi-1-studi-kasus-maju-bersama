"""Fungsi bersama: path dan pembersihan data."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA / "transaksi_raw.csv", parse_dates=["tanggal", "jatuh_tempo", "tanggal_bayar"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Tahap data preparation (CRISP-DM): duplikasi, kategori tidak konsisten, nilai kosong."""
    df = df.drop_duplicates(subset="id_transaksi").copy()
    df["kategori"] = df["kategori"].str.strip().str.title()
    df["diskon_persen"] = df["diskon_persen"].fillna(0.0)
    return df.reset_index(drop=True)
