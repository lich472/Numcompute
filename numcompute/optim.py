import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """
    Estimate gradient of a scalar function using finite differences.

    Parameters
    ----------
    f : callable
        Function that takes x and returns a scalar.
    x : array-like of shape (n,)
        Input vector.
    h : float
        Step size.
    method : str
        'central' or 'forward'

    Returns
    -------
    np.ndarray of shape (n,)
        Gradient vector.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")

    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_step = x.copy()

        if method == "forward":
            x_step[i] += h
            grad[i] = (f(x_step) - f(x)) / h

        elif method == "central":
            x_step_forward = x.copy()
            x_step_backward = x.copy()

            x_step_forward[i] += h
            x_step_backward[i] -= h

            grad[i] = (f(x_step_forward) - f(x_step_backward)) / (2 * h)

        else:
            raise ValueError("method must be 'central' or 'forward'")

    return grad

def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate Jacobian matrix of a vector function using finite differences.

    Parameters
    ----------
    F : callable
        Function that takes x and returns a vector (array-like).
    x : array-like of shape (n,)
        Input vector.
    h : float
        Step size.
    method : str
        'central' or 'forward'

    Returns
    -------
    np.ndarray of shape (m, n)
        Jacobian matrix where m = output dimension, n = input dimension.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")

    fx = np.asarray(F(x), dtype=float)

    if fx.ndim != 1:
        raise ValueError("F(x) must return a 1D array.")

    m = fx.size
    n = x.size

    J = np.zeros((m, n))

    for i in range(n):
        if method == "forward":
            x_step = x.copy()
            x_step[i] += h
            J[:, i] = (np.asarray(F(x_step)) - fx) / h

        elif method == "central":
            x_forward = x.copy()
            x_backward = x.copy()

            x_forward[i] += h
            x_backward[i] -= h

            J[:, i] = (
                np.asarray(F(x_forward)) - np.asarray(F(x_backward))
            ) / (2 * h)

        else:
            raise ValueError("method must be 'central' or 'forward'")

    return J