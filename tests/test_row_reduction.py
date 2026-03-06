import pytest

import numpy as np

from linalg.row_reduction import swap_rows, MatrixNxM


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
    assert consistent_augmented_matrix[0][0] == 2
    assert consistent_augmented_matrix[0][1] == -3
    assert consistent_augmented_matrix[0][2] == 2
    assert consistent_augmented_matrix[0][3] == 14
    assert consistent_augmented_matrix[1][0] == 1
    assert consistent_augmented_matrix[1][1] == 2
    assert consistent_augmented_matrix[1][2] == 3
    assert consistent_augmented_matrix[1][3] == 6
    swap_rows(consistent_augmented_matrix, 0, 1)
    assert consistent_augmented_matrix[1][0] == 2
    assert consistent_augmented_matrix[1][1] == -3
    assert consistent_augmented_matrix[1][2] == 2
    assert consistent_augmented_matrix[1][3] == 14
    assert consistent_augmented_matrix[0][0] == 1
    assert consistent_augmented_matrix[0][1] == 2
    assert consistent_augmented_matrix[0][2] == 3
    assert consistent_augmented_matrix[0][3] == 6
