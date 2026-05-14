from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CCPP_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/294/combined%2Bcycle%2Bpower%2Bplant.zip"
)
CCPP_XLSX_IN_ZIP = "CCPP/Folds5x2_pp.xlsx"

AI4I_CSV_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)

# Default retail/industrial blended tariff (₹/kWh) — editable in UI
DEFAULT_TARIFF_PEAK_INR_PER_KWH = 12.0
DEFAULT_TARIFF_OFFPEAK_INR_PER_KWH = 7.5

# MW → kWh for one hour (plant dataset is hourly averages)
MW_TO_KWH_PER_HOUR = 1000.0
