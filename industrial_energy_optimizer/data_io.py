"""Download and cache UCI datasets (Combined Cycle Power Plant, AI4I 2020)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

from industrial_energy_optimizer.config import (
    AI4I_CSV_URL,
    BASE_DIR,
    CCPP_XLSX_IN_ZIP,
    CCPP_ZIP_URL,
    DATA_DIR,
)

LOCAL_AI4I_CSV = (
    BASE_DIR / "ai4i+2020+predictive+maintenance+dataset" / "ai4i2020.csv"
)
LOCAL_CCPP_XLSX = BASE_DIR / "combined+cycle+power+plant" / "CCPP" / "Folds5x2_pp.xlsx"


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)


def _extract_ccpp_xlsx(zip_path: Path, dest_xlsx: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        target = None
        for n in names:
            if n.endswith("Folds5x2_pp.xlsx"):
                target = n
                break
        if target is None:
            target = CCPP_XLSX_IN_ZIP if CCPP_XLSX_IN_ZIP in names else None
        if target is None:
            raise FileNotFoundError(
                f"No Folds5x2_pp.xlsx in archive; entries: {names[:20]}..."
            )
        with zf.open(target) as zf_inner:
            dest_xlsx.write_bytes(zf_inner.read())


def load_ccpp(force_download: bool = False) -> pd.DataFrame:
    zip_path = DATA_DIR / "combined_cycle_power_plant.zip"
    xlsx_path = DATA_DIR / "Folds5x2_pp.xlsx"
    if LOCAL_CCPP_XLSX.exists() and not force_download:
        df = pd.read_excel(LOCAL_CCPP_XLSX, engine="openpyxl")
    else:
        if force_download or not zip_path.exists():
            _download(CCPP_ZIP_URL, zip_path)
        if force_download or not xlsx_path.exists():
            _extract_ccpp_xlsx(zip_path, xlsx_path)
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_ai4i(force_download: bool = False) -> pd.DataFrame:
    csv_path = DATA_DIR / "ai4i2020.csv"
    if LOCAL_AI4I_CSV.exists() and not force_download:
        return pd.read_csv(LOCAL_AI4I_CSV)
    if force_download or not csv_path.exists():
        _download(AI4I_CSV_URL, csv_path)
    return pd.read_csv(csv_path)
