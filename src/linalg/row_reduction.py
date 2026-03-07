from typing import Any

import numpy as np

MatrixNxM = np.ndarray[tuple[Any, Any]]


def swap_rows(m: MatrixNxM, i: int, j: int):
    temp = m[i].copy()
    m[i] = m[j]
    m[j] = temp


def scale_row(m: MatrixNxM, i: int, scaler: float | int):
    m[i] = m[i] * scaler


def replace_row(
    m: MatrixNxM, replace_idx: int, scaled_idx: int, scaler: float | int
):
    m[replace_idx] = m[replace_idx] + scaler * m[scaled_idx]
