from ..model.em import ensemble_learning, ensemble_learning_rsCV

def perform_em(data, label):
    ensemble_learning(data, label)
    ensemble_learning_rsCV(data, label)