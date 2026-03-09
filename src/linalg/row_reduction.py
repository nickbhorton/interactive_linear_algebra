from typing import Any

from math import isclose

import numpy as np

MatrixMxN = np.ndarray[tuple[Any, Any]]


def swap_rows(m: MatrixMxN, row1_idx: int, row2_idx: int):
    temp = m[row1_idx].copy()
    m[row1_idx] = m[row2_idx]
    m[row2_idx] = temp


def scale_row(m: MatrixMxN, row_idx: int, scaler: float | int):
    m[row_idx] = m[row_idx] * scaler


def replace_row(
    m: MatrixMxN, replace_idx: int, scaled_idx: int, scaler: float | int
):
    m[replace_idx] = m[replace_idx] + scaler * m[scaled_idx]


def find_next_pivot_bounds(m: MatrixMxN) -> tuple[int, int] | None:
    pivot_row: int = 0
    pivot_column: int = 0
    for column_idx in np.arange(m.shape[1]):
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
            return (pivot_row, pivot_column)
    return None


def get_pivot_one(m: MatrixMxN, pivot_bounds: tuple[int, int]):
    pivot_row, pivot_column = pivot_bounds
    for row_idx in np.arange(pivot_row, m.shape[0]):
        if not isclose(m[pivot_row, pivot_column], 0):
            scale_row(m, pivot_row, 1.0 / m[pivot_row, pivot_column])
            return
        else:
            swap_rows(m, pivot_row, row_idx + 1)


def clear_pivot_column(m: MatrixMxN, pivot_bounds: tuple[int, int]):
    pivot_row, pivot_column = pivot_bounds
    for row_idx in np.arange(pivot_row + 1, m.shape[0]):
        if not isclose(m[row_idx, pivot_column], 0):
            replace_row(m, row_idx, pivot_row, -m[row_idx, pivot_column])


def rref(m: MatrixMxN):
    # ensure in RE form
    pivot_bounds_list: list[tuple[int, int]] = []
    for row_idx in np.arange(m.shape[0]):
        pivot_bounds = find_next_pivot_bounds(m)
        if pivot_bounds is None:
            break
        get_pivot_one(m, pivot_bounds)
        clear_pivot_column(m, pivot_bounds)
        pivot_bounds_list.append(pivot_bounds)
    # clear up for RRE form
    for pivot_row, pivot_column in pivot_bounds_list[::-1]:
        for row_idx in np.arange(pivot_row)[::-1]:
            replace_row(m, row_idx, pivot_row, -m[row_idx, pivot_column])
