import numpy as np
import os
import csv

with open(os.path.join(os.path.dirname(__file__), 'ForcingHistory.csv')) as f:
           reader = csv.reader(f)
           header = next(reader)  # Skip header row if present
           ForcingHistory = np.array([row for row in reader]) # Reads CSV data
year = ForcingHistory[:,0].astype(float)
forcing = ForcingHistory[:,19].astype(float)
#print(year)
#print(forcing)
print(len(year))