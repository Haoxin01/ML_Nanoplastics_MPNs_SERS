import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


df_ud = pd.read_csv('../../data_reference/original_data/undetected groups.csv', index_col=0)
df_pe = pd.read_csv('../../data_reference/original_data/PE.csv', index_col=0)
df_pla = pd.read_csv('../../data_reference/original_data/PLA.csv', index_col=0)
df_pmma = pd.read_csv('../../data_reference/original_data/PMMA.csv', index_col=0)
df_ps = pd.read_csv('../../data_reference/original_data/PS.csv', index_col=0)

df_pe = df_pe.loc[df_pe['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
df_pla = df_pla.loc[df_pla['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
df_pmma = df_pmma.loc[df_pmma['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
df_ps = df_ps.loc[df_ps['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
df_ud = df_ud.loc[df_ud['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]

