#tools.py
import numpy as np
import xarray as xr
import os
import csv
#import scipy.io.netcdf

def csv_reader(file_name, coord_index=0, value_index=-1, skip_header=True):
    # Finds the file with name file_name within the Datafiler folder
    with open(os.path.join(os.path.dirname(__file__), 'Datafiler', file_name)) as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)  # Skips header row if skip_header is True
        data = np.array([row for row in reader])
    return data[:,coord_index].astype(float), data[:,value_index].astype(float)

def netcdf_reader(file_name):
    file_path = os.path.join(os.path.dirname(__file__), 'Datafiler', file_name)
    data = xr.open_dataset(file_path,engine= "scipy")
    return data

def warn(msg):
    """
    Print a formatted warning message in bold yellow with a ⚠️ symbol.
    Works in most Unix shells and modern PowerShell.
    """
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"{YELLOW_BOLD}⚠️  Warning:{RESET} {msg}")