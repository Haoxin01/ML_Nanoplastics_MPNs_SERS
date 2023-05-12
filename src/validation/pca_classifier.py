from ..model.pca import dim_reduction, pca_visualization

def perform_pca(data):
    principalDf = dim_reduction(data)
    pca_visualization(principalDf, 'data after PCA')