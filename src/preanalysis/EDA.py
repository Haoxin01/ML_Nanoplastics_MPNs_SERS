import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Correlation heatmap use matplotlib
def correlation_heatmap(df):
    """
    Plot a correlation heatmap for the given dataframe.
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(df.corr(), cmap='coolwarm', interpolation='nearest')
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fig.colorbar(im)
    # set title
    plt.title('Correlation Heatmap for Undetected')
    plt.show()
    # save
    fig.savefig('./result/EDA/correlation_heatmap_UD.png')


df = pd.read_csv('../data/test/undetected groups.csv', index_col=0)

# See the first 5 rows of the data
print(df.head())

# Get a summary of the data
print(df.info())

# Get basic statistics
print(df.describe())
# save
df.describe().to_csv('./result/EDA/PE_describe.csv')

correlation_heatmap(df.drop(df.columns[0], axis=1))

# save correlation matrix
df.drop(df.columns[0], axis=1).corr().to_csv('./result/EDA/correlation_matrix.csv')



if __name__ == '__main__':
    pass