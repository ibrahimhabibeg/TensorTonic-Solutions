import numpy as np
from scipy.special import comb
from functools import reduce

def binomial_pmf(n, p, k):
    return comb(n, k) * (p**k) * ((1-p) ** (n-k))

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    return binomial_pmf(n, p, k), reduce(lambda x, y: x+y, [binomial_pmf(n, p, i) for i in range(k+1)], 0) 