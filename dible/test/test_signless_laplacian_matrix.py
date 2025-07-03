import numpy as np
import pandas as pd
from dible import blockencoding, tools
from mindquantum import *


def signless_laplacian_matrix(num_vertices, alpha1, alpha2, alpha3):
    """
    Constructs a signless Laplacian matrix for a circular graph.

    Args:
        num_vertices (int): The number of vertices in the circular graph.
            It should be a positive integer.
        alpha1 (float): The value for the diagonal elements of the matrix.
        alpha2 (float): The value for the right neighbor elements of the matrix.
        alpha3 (float): The value for the left neighbor elements of the matrix.

    Returns:
        numpy.array: The signless Laplacian matrix of the circular graph.
    """
    matrix = np.zeros((num_vertices, num_vertices), dtype=complex)
    for j in range(num_vertices):
        matrix[j][j] = alpha1
        if j != num_vertices - 1:
            matrix[j + 1][j] = alpha2
        else:
            matrix[0][j] = alpha2
        if j != 0:
            matrix[j - 1][j] = alpha3
        else:
            matrix[num_vertices - 1][j] = alpha3

    return matrix


def get_data_item(alpha1, alpha2, alpha3, dim):
    """
    Generates the dictionary for the signless Laplacian matrix.

    Args:
        alpha1 (float): Non-zero element in the matrix.
        alpha2 (float): Non-zero element in the matrix.
        alpha3 (float): Non-zero element in the matrix.
        dim (int): The dimension of the matrix, which must be a power of two.

    Returns:
        dict: The dictionary consisting of data items.
    """
    n = int(np.log2(dim))
    data_item = {
        0: [alpha1,
            0,
            tools.binary_range(0, dim - 1, n, True, True)],
        1: [alpha2,
            1,
            tools.binary_range(0, dim - 1, n, True, True)],
        2: [alpha3,
            -1,
            tools.binary_range(0, dim - 1, n, True, True)]
    }

    return data_item


def test_signless_laplacian_matrix(data_item, dim):
    """
    Tests the construction of block encoding a signless Laplacian matrix.

    Args:
        data_item (dict): The dictionary consisting of data items.
        dim (int): The dimension of the signless Laplacian matrix, which must be a power of two.

    Returns:
        tuple: A tuple containing the constructed quantum circuit and the encoded signless Laplacian matrix.
            - circuit: The quantum circuit of block encoding.
            - encoded_matrix: The encoded signless Laplacian matrix.
    """
    # The number of qubits of register idx
    num_idx_qubits = tools.num_qubits(len(data_item))

    # The number of working qubits
    num_working_qubits = tools.num_qubits(dim)

    # The number of qubits of circuit
    num_qubits = num_idx_qubits + 1 + num_working_qubits

    # Sparse block encoding
    circuit = blockencoding.qcircuit(data_item=data_item,
                                     num_working_qubits=num_working_qubits)

    # Get the encoded matrix
    encoded_matrix = blockencoding.get_encoded_matrix(circuit, num_qubits, num_working_qubits)

    return circuit, encoded_matrix


if __name__ == '__main__':
    alpha1 = 1
    alpha2 = 2
    alpha3 = 3
    num_vertices = 8

    # Get data items
    data_items = get_data_item(alpha1, alpha2, alpha3, dim=num_vertices)

    # Get the signless Laplacian matrix to be encoded
    matrix = signless_laplacian_matrix(num_vertices, alpha1, alpha2, alpha3)

    # Get the circuit and the encoded signless Laplacian matrix
    circuit, encoded_matrix = test_signless_laplacian_matrix(data_items, dim=num_vertices)

    # Compute the subnormalization
    subnormalization = abs(alpha1) + abs(alpha2) + abs(alpha3)

    print(circuit)
    matrix_pd = pd.DataFrame(matrix)
    matrix_pd.to_excel('signless_Laplacian_matrix.xlsx')
    encoded_matrix_pd = pd.DataFrame(encoded_matrix)
    encoded_matrix_pd.to_excel('signless_Laplacian_matrix_encoded.xlsx')
    print('Actual subnormalization:')
    print(np.linalg.norm(matrix) / np.linalg.norm(encoded_matrix))
    print('Theoretical subnormalization:')
    print(subnormalization)
    error = np.linalg.norm(matrix - subnormalization * encoded_matrix)
    print('The error of block encoding:')
    print(error)
