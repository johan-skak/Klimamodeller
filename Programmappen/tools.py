import numpy as np
import xarray as xr
import os
import csv
#import scipy.io.netcdf

def csv_reader(file_name, skip_header=True):
    # Finds the file with name file_name within the Datafiler folder
    with open(os.path.join(os.path.dirname(__file__), 'Datafiler', file_name)) as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)  # Skips header row if skip_header is True
        data = np.array([row for row in reader])
    return data

def netcdf_reader(file_name):
    with open(os.path.join(os.path.dirname(__file__), 'Datafiler', file_name)) as f:
       data = xr.open_dataset(f,engine= "scipy")
    return data

def warn(msg):
    """
    Print a formatted warning message in bold yellow with a ⚠️ symbol.
    Works in most Unix shells and modern PowerShell.
    """
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"{YELLOW_BOLD}⚠️  Warning:{RESET} {msg}")