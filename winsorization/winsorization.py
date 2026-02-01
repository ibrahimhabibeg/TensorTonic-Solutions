import math

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    N = len(values)
    
    lower_val_idx = (N-1) * lower_pct / 100
    lower_val_remainder = lower_val_idx % 1
    lower_val_idx = math.floor(lower_val_idx)
    lower_val = values[lower_val_idx] + (values[lower_val_idx + 1] - values[lower_val_idx]) * lower_val_remainder if lower_val_idx < N-1 else values[lower_val_idx]

    upper_val_idx = (N-1) * upper_pct / 100
    upper_val_remainder = upper_val_idx % 1
    upper_val_idx = math.floor(upper_val_idx)
    upper_val = values[upper_val_idx] + (values[upper_val_idx + 1] - values[upper_val_idx]) * upper_val_remainder if upper_val_idx < N-1 else values[upper_val_idx]

    values = [max(v, lower_val) for v in values]
    values = [min(v, upper_val) for v in values]
    return values
    