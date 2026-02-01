import numpy as np

def calc_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def calc_precision(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0.0

def calc_recall(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 0.0

def calc_f1(y_true, y_pred):
    precision = calc_precision(y_true, y_pred)
    recall = calc_recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

def calc_mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()

def calc_rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5

def calc_precision_at_3(y_true, y_pred):
    ranking = np.argsort(-y_pred)
    is_in_top_3 = ranking < 3
    return (is_in_top_3 & y_true).sum() / 3

def calc_recall_at_3(y_true, y_pred):
    ranking = np.argsort(-y_pred)
    is_in_top_3 = ranking < 3
    return (is_in_top_3 & y_true).sum() / y_true.sum() if y_true.sum() > 0 else 0.0
    
def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    assert system_type in ["classification", "regression", "ranking"]

    # Use numpy for vectorization
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Return Values
    if system_type == "classification":
        return [("accuracy", calc_accuracy(y_true, y_pred)), ("f1", calc_f1(y_true, y_pred)), ("precision", calc_precision(y_true, y_pred)), ("recall", calc_recall(y_true, y_pred))]

    if system_type == "regression":
        return [("mae", calc_mae(y_true, y_pred)), ("rmse", calc_rmse(y_true, y_pred))]

    if system_type == "ranking":
        return [("precision_at_3", calc_precision_at_3(y_true, y_pred)), ("recall_at_3", calc_recall_at_3(y_true, y_pred))]


