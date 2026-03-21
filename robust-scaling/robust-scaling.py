def calc_median(sorted_values):
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2.0
    else:
        return sorted_values[n//2]
    
def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    n = len(values)
    sorted_values = sorted(values)
    if n == 1:
        return [0.0]
    median = calc_median(sorted_values)
    Q1 = calc_median(sorted_values[:n//2])
    Q3 = calc_median(sorted_values[(1 + n)//2:])
    # return [Q1, median, Q3]
    if Q1 == Q3:
        return [v - median for v in values]
    else:
        return [(v - median) / (Q3-Q1) for v in values]