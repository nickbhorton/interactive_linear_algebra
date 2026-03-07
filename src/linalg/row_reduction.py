from typing import Any

from math import isclose

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


def find_next_pivot_column(m: MatrixNxM) -> int:
    pivot_column: int = 0
    pivot_row: int = 0
    for column_idx in np.arange(m.shape[1] - 1):
        # if the column is all zeros below last pivot, goto next column
        if np.allclose(
            m[pivot_row:, column_idx], np.zeros(len(m[pivot_row:, column_idx]))
        ):
            pivot_column += 1
            continue
        # if the column is a pivot column, goto next column and row
        elif np.allclose(
            m[pivot_row + 1 :, column_idx],
            np.zeros(len(m[pivot_row + 1 :, column_idx])),
        ) and isclose(m[pivot_row, column_idx], 1):
            pivot_column += 1
            pivot_row += 1
        else:
            return pivot_column
    raise Exception("matrix is inconsistent")


def row_reduce(m: MatrixNxM):
    pass
