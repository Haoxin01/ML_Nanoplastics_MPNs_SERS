import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 定义三种函数模型
def logistic(x, a, b, c):
    return a / (1 + np.exp(-c * (x - b)))

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

# 定义拟合并画图函数
def fit_and_plot(plastic, xdata, ydata, func, func_name, yerr):
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    x_continuous = np.linspace(min(xdata), max(xdata), 500)
    y_pred_continuous = func(x_continuous, *popt)
    y_pred = func(xdata, *popt)
    r2 = r2_score(ydata, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_continuous, y_pred_continuous, label=f'{func_name}', linewidth=3.5)
    ax.scatter(xdata, ydata, color='black')
    ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', color='black')
    ax.set_xlabel('Concentration (ppm)', fontsize=22, weight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=22, weight='bold')
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    plt.show()
    print(f'Fitted parameters: {popt}')
    print(f'R2 score: {r2:.4f}')

# 定义数据处理函数
def process_data(plastic):
    df = pd.read_csv(f'D:\\Nanoplastics-ML\\data\\regression_data\\{plastic}.csv', header=None)
    df = df.T
    df.columns = ['Concentration', 'Intensity']
    df_group = df.groupby('Concentration').agg(['mean', 'std']).reset_index()
    xdata = df_group['Concentration'].values
    ydata = df_group[('Intensity', 'mean')].values
    yerr = df_group[('Intensity', 'std')].values
    return xdata, ydata, yerr

# 主程序
plastics = ['PS', 'PE', 'PMMA', 'PLA']
functions = {'logistic': logistic, 'exponential': exponential, 'polynomial': polynomial}

for plastic in plastics:
    xdata, ydata, yerr = process_data(plastic)
    for func_name, func in functions.items():
        fit_and_plot(plastic, xdata, ydata, func, func_name, yerr)