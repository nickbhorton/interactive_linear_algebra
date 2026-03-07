import pytest

import numpy as np

from math import isclose

from linalg.row_reduction import (
    swap_rows,
    scale_row,
    replace_row,
    find_next_pivot_bounds,
    get_pivot_one,
    clear_pivot_column,
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
    assert find_next_pivot_bounds(e1) == (0, 0)
    e2 = np.array(
        [
            [0, 2, 3, 6],
            [0, -3, 2, 14],
            [0, 1, -1, 2],
        ]
    )
    assert find_next_pivot_bounds(e2) == (0, 1)
    e3 = np.array(
        [
            [1, 2, 3, 6],
            [0, -3, 2, 14],
            [0, 1, -1, 2],
        ]
    )
    assert find_next_pivot_bounds(e3) == (1, 1)
    e4 = np.array(
        [
            [0, 2, 3, 6],
            [1, -3, 2, 14],
            [0, 1, -1, 2],
        ]
    )
    assert find_next_pivot_bounds(e4) == (0, 0)
    e5 = np.array(
        [
            [1, 2, 3, 6],
            [0, 0, 2, 14],
            [0, 0, -1, 2],
        ]
    )
    assert find_next_pivot_bounds(e5) == (1, 2)
    e6 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2, 14],
            [0, 0, 0, 2],
        ]
    )
    assert find_next_pivot_bounds(e6) == (2, 3)
    e7 = np.array(
        [
            [1, 2, 3, 6],
            [0, 0, 0, 14],
            [0, 0, 0, 2],
        ]
    )
    assert find_next_pivot_bounds(e7) == (1, 3)
    # e8 is in REF technically but calling find_next_pivot_bounds should error
    e8 = np.array(
        [
            [1, 2, 3, 6],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    with pytest.raises(Exception):
        find_next_pivot_bounds(e8)


def test_get_pivot_one():
    e1 = np.array(
        [
            [1, 2, 3, 6],
            [2, -3, 2, 14],
            [3, 1, -1, 2],
        ]
    )
    get_pivot_one(e1, find_next_pivot_bounds(e1))
    assert np.allclose(e1, e1)
    e2 = np.array(
        [
            [2, 2, 3, 6],
            [2, -3, 2, 14],
            [3, 1, -1, 2],
        ],
        dtype=float,
    )
    a2 = np.array(
        [
            [1, 1, 1.5, 3],
            [2, -3, 2, 14],
            [3, 1, -1, 2],
        ],
        dtype=float,
    )
    get_pivot_one(e2, find_next_pivot_bounds(e2))
    assert np.allclose(e2, a2)
    e3 = np.array(
        [
            [1, 2, 3, 6],
            [0, -3, 2, 14],
            [0, 1, -1, 2],
        ],
        dtype=float,
    )
    a3 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2 / -3, 14 / -3],
            [0, 1, -1, 2],
        ],
        dtype=float,
    )
    get_pivot_one(e3, find_next_pivot_bounds(e3))
    assert np.allclose(e3, a3)
    e4 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2, 14],
            [0, 0, -1, 2],
        ],
        dtype=float,
    )
    a4 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2, 14],
            [0, 0, 1, -2],
        ],
        dtype=float,
    )
    get_pivot_one(e4, find_next_pivot_bounds(e4))
    assert np.allclose(e4, a4)
    e5 = np.array(
        [
            [1, 2, 3, 6, 8],
            [0, 0, 1, 14, 10],
            [0, 0, 0, 2, 14],
        ],
        dtype=float,
    )
    a5 = np.array(
        [
            [1, 2, 3, 6, 8],
            [0, 0, 1, 14, 10],
            [0, 0, 0, 1, 7],
        ],
        dtype=float,
    )
    get_pivot_one(e5, find_next_pivot_bounds(e5))
    assert np.allclose(e5, a5)
    e6 = np.array(
        [
            [0, 2, 3, 6],
            [2, -3, 2, 14],
            [3, 1, -1, 2],
        ],
        dtype=float,
    )
    a6 = np.array(
        [
            [1, -3 / 2, 1, 7],
            [0, 2, 3, 6],
            [3, 1, -1, 2],
        ],
        dtype=float,
    )
    get_pivot_one(e6, find_next_pivot_bounds(e6))
    assert np.allclose(e6, a6)
    e7 = np.array(
        [
            [0, 2, 3, 6],
            [0, -3, 2, 14],
            [3, 1, -1, 2],
        ],
        dtype=float,
    )
    a7 = np.array(
        [
            [1, 1 / 3, -1 / 3, 2 / 3],
            [0, 2, 3, 6],
            [0, -3, 2, 14],
        ],
        dtype=float,
    )
    get_pivot_one(e7, find_next_pivot_bounds(e7))
    assert np.allclose(e7, a7)
    e8 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2, 14],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
        ],
        dtype=float,
    )
    a8 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2, 14],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    get_pivot_one(e8, find_next_pivot_bounds(e8))
    assert np.allclose(e8, a8)


def test_clear_pivot_column():
    e1 = np.array(
        [
            [0, -7, -4, 2],
            [2, 4, 6, 12],
            [3, 1, -1, -2],
        ]
    )
    a1 = np.array(
        [
            [1, 2, 3, 6],
            [0, -7, -4, 2],
            [0, -5, -10, -20],
        ]
    )
    pivot_bounds = find_next_pivot_bounds(e1)
    get_pivot_one(e1, pivot_bounds)
    clear_pivot_column(e1, pivot_bounds)
    assert np.allclose(e1, a1)
    e2 = np.array(
        [
            [1, 2, 3, 6],
            [0, -5, -10, -20],
            [0, -7, -4, 2],
        ]
    )
    a2 = np.array(
        [
            [1, 2, 3, 6],
            [0, 1, 2, 4],
            [0, 0, 10, 30],
        ]
    )
    pivot_bounds = find_next_pivot_bounds(e2)
    get_pivot_one(e2, pivot_bounds)
    clear_pivot_column(e2, pivot_bounds)
    assert np.allclose(e2, a2)
