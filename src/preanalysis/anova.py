import scipy.stats as stats
import pandas as pd

# read data_reference
df_ud = pd.read_csv('../../data_reference/original_data/undetected groups.csv', index_col=0)
df_pe = pd.read_csv('../../data_reference/original_data/PE.csv', index_col=0)
df_pla = pd.read_csv('../../data_reference/original_data/PLA.csv', index_col=0)
df_pmma = pd.read_csv('../../data_reference/original_data/PMMA.csv', index_col=0)
df_ps = pd.read_csv('../../data_reference/original_data/PS.csv', index_col=0)

def anova_pe(df):
    df_pe = pd.read_csv('../../data_reference/original_data/PE.csv', index_col=0)

def anova_pla(df):
    df_pla = pd.read_csv('../../data_reference/original_data/PLA.csv', index_col=0)

def anova_pmma(df):
    df_pmma = pd.read_csv('../../data_reference/original_data/PMMA.csv', index_col=0)

def anova_ps(df):
    df_ps = pd.read_csv('../../data_reference/original_data/PS.csv', index_col=0)

def anova_test(df_ud):
    pass

# 551.15, 869.87, 998.37, 1134.67
# get rows where wavenumber is 551.15, 869.87, 998.37, 1134.67
# df_pe = df_pe.loc[df_pe['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
# df_pla = df_pla.loc[df_pla['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
# df_pmma = df_pmma.loc[df_pmma['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
# df_ps = df_ps.loc[df_ps['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]
# df_ud = df_ud.loc[df_ud['Wavenumber'].isin([551.15, 869.87, 998.37, 1134.67])]

# fvalue, pvalue = stats.f_oneway(df_pe['PE0.05ppm-1'], df_pe['PE0.05ppm-2'], df_pe['PE0.05ppm-3'],
#                                 df_pe['PE0.1ppm-1'], df_pe['PE0.1ppm-2'], df_pe['PE0.1ppm-3'], df_pe['PE0.5ppm-1'],
#                                 df_pe['PE0.5ppm-2'], df_pe['PE0.5ppm-3'], df_pe['PE1ppm-1'], df_pe['PE1ppm-2'],
#                                 df_pe['PE1ppm-3'], df_pe['PE5ppm-1'], df_pe['PE5ppm-2'], df_pe['PE5ppm-3'],
#                                 df_pe['PE10ppm-1'], df_pe['PE10ppm-2'], df_pe['PE10ppm-3'], df_pe['PE50ppm-1'],
#                                 df_pe['PE50ppm-2'], df_pe['PE50ppm-3'], df_pe['PE100ppm-1'], df_pe['PE100ppm-2'],
#                                 df_pe['PE100ppm-3'], df_pe['PE200ppm-1'], df_pe['PE200ppm-2'], df_pe['PE200ppm-3'])

fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.05ppm-1'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.05ppm-2'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.05ppm-3'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.1ppm-1'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.1ppm-2'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.1ppm-3'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.5ppm-1'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.5ppm-2'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE0.5ppm-3'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE1ppm-1'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE1ppm-2'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE1ppm-3'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE5ppm-1'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE5ppm-2'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE5ppm-3'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE10ppm-1'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE10ppm-2'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(df_ud['PS0ppm-1'], df_pe['PE10ppm-3'])
print(fvalue, pvalue)




print("F-value:", fvalue)
print("P-value:", pvalue)
