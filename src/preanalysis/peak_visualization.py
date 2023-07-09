import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def whole_peak_visualization(peak_df, name):
    """
    Visualize the whole peak data_reference, every column is a line except the first column.
    """
    # drop the first column
    peak_df = peak_df.drop(peak_df.columns[0], axis=1)
    fig, ax = plt.subplots(figsize=(20, 20))
    for col in peak_df.columns:
        ax.plot(peak_df[col], label=col)
    ax.legend()
    plt.show()
    # save
    fig.savefig('./result/peak_vis/' + name + '.png')


if __name__ == '__main__':
    peak_df = pd.read_csv('../../data_reference/original_data/undetected groups.csv', index_col=0)
    whole_peak_visualization(peak_df, 'unexpected_group_visualization')

    peak_df = pd.read_csv('../../data_reference/original_data/PE.csv', index_col=0)
    whole_peak_visualization(peak_df, 'PE_visualization')

    peak_df = pd.read_csv('../../data_reference/original_data/PLA.csv', index_col=0)
    whole_peak_visualization(peak_df, 'PLA_visualization')

    peak_df = pd.read_csv('../../data_reference/original_data/PMMA.csv', index_col=0)
    whole_peak_visualization(peak_df, 'PMMA_visualization')

    peak_df = pd.read_csv('../../data_reference/original_data/PS.csv', index_col=0)
    whole_peak_visualization(peak_df, 'PS_visualization')