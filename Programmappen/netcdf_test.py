import xarray as xr
import os
def netcdf_reader(file_name):
    file_path = os.path.join(os.path.dirname(__file__), 'Datafiler', file_name)
    data = xr.open_dataset(file_path,engine= "scipy")
    return data
data = netcdf_reader('temperature_history.nc')
print(data)