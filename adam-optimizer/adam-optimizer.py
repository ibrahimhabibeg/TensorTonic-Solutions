import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - (beta1 ** t))
    v_hat = v_new / (1 - (beta2 ** t))
    param = param - lr * (m_hat / ((v_hat **0.5) + eps))
    return param, m_new, v_new