"""
Download the Online Retail II dataset.

Default source is the UCI Machine Learning Repository (no auth needed). This is
the same UK online-retail transactions dataset published on Kaggle as
"Online Retail II UCI".

    python download_data.py

--- Using Kaggle instead -----------------------------------------------------
If you'd rather pull it through the Kaggle API, place your kaggle.json in
~/.kaggle/ (or the project root) and run:

    pip install kaggle
    kaggle datasets download -d mashlyn/online-retail-ii-uci -p data --unzip

then point the pipeline at the resulting file.
"""
import zipfile
from pathlib import Path
import urllib.request

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_PATH = DATA_DIR / "online_retail_II.zip"
XLSX_PATH = DATA_DIR / "online_retail_II.xlsx"


def main():
    if XLSX_PATH.exists():
        print(f"Already present: {XLSX_PATH}")
        return
    print(f"Downloading from {URL} ...")
    temporary = ZIP_PATH.with_suffix(".part")
    urllib.request.urlretrieve(URL, temporary)
    temporary.replace(ZIP_PATH)
    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = [n for n in z.namelist() if Path(n).name == XLSX_PATH.name]
        if len(members) != 1:
            raise ValueError("Expected exactly one official workbook in archive")
        XLSX_PATH.write_bytes(z.read(members[0]))
    print(f"Done -> {XLSX_PATH}")


if __name__ == "__main__":
    main()
