#utils.py
import re
import pandas as pd
from pathlib import Path
from colorama import Fore, Style


def open_file(file_name: str, path: str = "") -> pd.DataFrame:
    full_path = Path(path) / file_name
    if not full_path.is_file():
        print(f" {Fore.RED}[NOT FOUND]{Style.RESET_ALL} File : {file_name}")
        return None
    else:
        print(f"{Fore.LIGHTGREEN_EX}[FOUND]{Style.RESET_ALL} File : {file_name}")
        print(f"\tOpening {file_name} dataset")
        data = pd.read_csv(full_path)
        print(
            "\tFile loaded with :",
            str(data.shape[0]),
            "lines and",
            str(data.shape[1]),
            "columns",
        )
        return data

def save_file(data: pd.DataFrame, file_name: str, path: str = ".") -> bool:
    output_path = Path(path) / file_name
    try:
        data.to_csv(output_path, index=False)
        print(
            f"{Fore.LIGHTGREEN_EX}[SAVED]{Style.RESET_ALL} File : {file_name} in {path}"
        )
        return True
    except Exception as e:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL} Saving File: {file_name}, Error: {e}"
        )
        return False