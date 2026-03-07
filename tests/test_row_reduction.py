import pytest

import numpy as np

from math import isclose

from linalg.row_reduction import (
    swap_rows,
    scale_row,
    replace_row,
    find_next_pivot_column,
    MatrixNxM,
)


@pytest.fixture
def consistent_augmented_matrix() -> MatrixNxM:
    return np.array(
        [
            [1, 2, 3, 6],
            [2, -3, 2, 14],
            [3, 1, -1, 2],
        ]
    )


def test_swap_rows(consistent_augmented_matrix):
    swap_rows(consistent_augmented_matrix, 0, 1)
    assert isclose(consistent_augmented_matrix[0][0], 2)
    assert isclose(consistent_augmented_matrix[0][1], -3)
    assert isclose(consistent_augmented_matrix[0][2], 2)
    assert isclose(consistent_augmented_matrix[0][3], 14)
    assert isclose(consistent_augmented_matrix[1][0], 1)
    assert isclose(consistent_augmented_matrix[1][1], 2)
    assert isclose(consistent_augmented_matrix[1][2], 3)
    assert isclose(consistent_augmented_matrix[1][3], 6)
    swap_rows(consistent_augmented_matrix, 0, 1)
    assert isclose(consistent_augmented_matrix[1][0], 2)
    assert isclose(consistent_augmented_matrix[1][1], -3)
    assert isclose(consistent_augmented_matrix[1][2], 2)
    assert isclose(consistent_augmented_matrix[1][3], 14)
    assert isclose(consistent_augmented_matrix[0][0], 1)
    assert isclose(consistent_augmented_matrix[0][1], 2)
    assert isclose(consistent_augmented_matrix[0][2], 3)
    assert isclose(consistent_augmented_matrix[0][3], 6)


def test_scale_row(consistent_augmented_matrix):
    scale_row(consistent_augmented_matrix, 0, 2)
    assert isclose(consistent_augmented_matrix[0][0], 2)
    assert isclose(consistent_augmented_matrix[0][1], 4)
    assert isclose(consistent_augmented_matrix[0][2], 6)
    assert isclose(consistent_augmented_matrix[0][3], 12)


def test_replace_row(consistent_augmented_matrix):
    replace_row(consistent_augmented_matrix, 1, 0, -2)
    assert isclose(consistent_augmented_matrix[1][0], 0)
    assert isclose(consistent_augmented_matrix[1][1], -7)
    assert isclose(consistent_augmented_matrix[1][2], -4)
    assert isclose(consistent_augmented_matrix[1][3], 2)


def test_find_next_pivot_column():
    e1 = np.array(
        [
            [1, 2, 3, 6],
            [2, -3, 2, 14],
            [3, 1, -1, 2],
        ]
    )
    assert find_next_pivot_column(e1) == 0
    e2 = np.array(
        [
            [0, 2, 3, 6],
            [0, -3, 2, 14],
            [0, 1, -1, 2],
        ]
    )
    assert find_next_pivot_column(e2) == 1
    e3 = np.array(
        [
            [1, 2, 3, 6],
            [0, -3, 2, 14],
            [0, 1, -1, 2],
        ]
    )
    assert find_next_pivot_column(e3) == 1
    e4 = np.array(
        [
            [0, 2, 3, 6],
            [1, -3, 2, 14],
            [0, 1, -1, 2],
        ]
    )
    assert find_next_pivot_column(e4) == 0
    e5 = np.array(
        [
            [1, 2, 3, 6],
            [0, 0, 2, 14],
            [0, 0, -1, 2],
        ]
    )
    assert find_next_pivot_column(e5) == 2
    e6 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2, 14],
            [0, 0, 0, 2],
        ]
    )
    with pytest.raises(Exception):
        find_next_pivot_column(e6)
    e7 = np.array(
        [
            [1, 2, 3, 6],
            [0, 0, 0, 14],
            [0, 0, 0, 2],
        ]
    )
    with pytest.raises(Exception):
        find_next_pivot_column(e7)
