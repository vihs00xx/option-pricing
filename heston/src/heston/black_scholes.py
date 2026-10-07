"""Black-Scholes in forward form (per unit strike), for the xi -> 0 limit check."""

import numpy as np
from scipy.special import ndtr


def bs_price(x, w, tau, r, call=True):
    """K e^{-r tau} [e^x N(d1) - N(d2)] with x = log(F/K) and total variance w tau; puts by parity."""
    x, w, tau, r, call = np.broadcast_arrays(*map(np.asarray, (x, w, tau, r, call)))
    sd = np.sqrt(w * tau)
    d1 = x / sd + 0.5 * sd
    d2 = d1 - sd
    disc = np.exp(-r * tau)
    c = disc * (np.exp(x) * ndtr(d1) - ndtr(d2))
    p = disc * (ndtr(-d2) - np.exp(x) * ndtr(-d1))
    return np.where(call, c, p)
