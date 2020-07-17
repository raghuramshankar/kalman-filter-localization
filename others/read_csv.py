import pandas as pd

AMZ_data = pd.read_csv('AMZ_data_resample_gps.csv')

print(len(AMZ_data.lat))