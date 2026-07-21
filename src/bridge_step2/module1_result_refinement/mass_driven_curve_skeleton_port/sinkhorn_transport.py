from __future__ import annotations

import numpy as np


def sinkhorn_transport(
    a: np.ndarray,
    b: np.ndarray,
    k_matrix: np.ndarray,
    u_matrix: np.ndarray,
    lambda_value: float,
    stopping_criterion: str = "marginalDifference",
    p_norm: float = np.inf,
    tolerance: float = 5e-3,
    max_iter: int = 5000,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    one_vs_n = a.shape[1] == 1
    if not one_vs_n and a.shape[1] != b.shape[1]:
        raise ValueError("a must be a column vector or have the same number of columns as b.")

    big_n = b.shape[1] > b.shape[0]

    some_zero_values = False
    if one_vs_n:
        support = a[:, 0] > 0
        if not np.all(support):
            some_zero_values = True
            k_matrix = k_matrix[support, :]
            u_matrix = u_matrix[support, :]
            a = a[support, :]
        ainv_k = k_matrix / a

    u = np.ones((a.shape[0], b.shape[1]), dtype=float) / a.shape[0]
    d_old = np.ones(b.shape[1], dtype=float) if stopping_criterion == "distanceRelativeDecrease" else None

    iteration = 0
    while iteration < max_iter:
        if one_vs_n:
            if big_n:
                u = 1.0 / (ainv_k @ (b / (k_matrix.T @ u)))
            else:
                u = 1.0 / (ainv_k @ (b / ((u.T @ k_matrix).T)))
        else:
            if big_n:
                u = a / (k_matrix @ (b / ((u.T @ k_matrix).T)))
            else:
                u = a / (k_matrix @ (b / (k_matrix.T @ u)))

        iteration += 1

        if iteration % 20 == 1 or iteration == max_iter:
            if big_n:
                v = b / (k_matrix.T @ u)
            else:
                v = b / ((u.T @ k_matrix).T)

            if one_vs_n:
                u = 1.0 / (ainv_k @ v)
            else:
                u = a / (k_matrix @ v)

            if stopping_criterion == "distanceRelativeDecrease":
                d_values = np.sum(u * (u_matrix @ v), axis=0)
                criterion = np.linalg.norm(d_values / d_old - 1.0, ord=p_norm)
                if criterion < tolerance or np.isnan(criterion):
                    break
                d_old = d_values
            elif stopping_criterion == "marginalDifference":
                criterion = np.linalg.norm(np.sum(np.abs(v * (k_matrix.T @ u) - b), axis=0), ord=p_norm)
                if criterion < tolerance or np.isnan(criterion):
                    break
            else:
                raise ValueError("Unsupported stopping criterion.")

            iteration += 1
            if verbose > 0:
                print(f"Iteration: {iteration} Criterion: {criterion}")
            if np.isnan(criterion):
                raise FloatingPointError("NaN encountered during Sinkhorn iteration.")

    if stopping_criterion == "marginalDifference":
        d_values = np.sum(u * (u_matrix @ v), axis=0)

    alpha = np.log(u)
    beta = np.log(v)
    beta[np.isneginf(beta)] = 0
    if one_vs_n:
        lower_bound = (a[:, 0].T @ alpha + np.sum(b * beta, axis=0)) / lambda_value
    else:
        alpha[np.isneginf(alpha)] = 0
        lower_bound = (np.sum(a * alpha, axis=0) + np.sum(b * beta, axis=0)) / lambda_value

    if some_zero_values:
        recovered_u = np.zeros((support.shape[0], b.shape[1]), dtype=float)
        recovered_u[support, :] = u
        u = recovered_u

    return d_values, lower_bound, u, v
