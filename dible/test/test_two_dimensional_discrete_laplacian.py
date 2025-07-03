import numpy as np
import pandas as pd
from dible import blockencoding, tools
from mindquantum import *


def get_non_zero_elements(parameters):
    """
    Computes all distinct non-zero elements of the two-dimensional discrete Laplacian.

    Args:
        parameters (dict): A dictionary containing the discretization parameters.
            - 'delta_x': The discretization step size in the x-direction.
            - 'delta_y': The discretization step size in the y-direction.

    Returns:
        list: A list of distinct non-zero elements [A0, A1, A2].
    """
    delta_x = parameters['delta_x']
    delta_y = parameters['delta_y']
    A0 = - 2 * (1 / delta_x ** 2 + 1 / delta_y ** 2)
    A1 = 1 / delta_x ** 2
    A2 = 1 / delta_y ** 2
    non_zero_elements = [A0, A1, A2]

    return non_zero_elements


# Get the two_dimensional discrete Laplacian
def two_dimensional_discrete_laplacian(non_zero_elements, nx, ny):
    """
    Constructs the two-dimensional discrete Laplacian.

    Args:
        non_zero_elements (list): A list of three non-zero elements [A0, A1, A2].
            - A0: The diagonal element.
            - A1: The horizontal off-diagonal element.
            - A2: The vertical off-diagonal element.
        nx (int): The number of grid points in the x-direction.
        ny (int): The number of grid points in the y-direction.

    Returns:
        numpy.array: The two-dimensional discrete Laplacian.
    """
    matrix = np.zeros((nx * ny, nx * ny))
    for i1 in range(nx):
        for i2 in range(nx):
            for j1 in range(ny):
                for j2 in range(ny):
                    if i1 == j1 and i2 == j2:
                        matrix[i1 + i2 * nx, j1 + j2 * ny] = non_zero_elements[0]
                    elif abs(i1 - j1) == 1 and i2 == j2:
                        matrix[i1 + i2 * nx, j1 + j2 * ny] = non_zero_elements[1]
                    elif abs(i2 - j2) == 1 and i1 == j1:
                        matrix[i1 + i2 * nx, j1 + j2 * ny] = non_zero_elements[2]

    return matrix


def get_data_item(non_zero_elements, nx, ny):
    """
    Constructs the dictionary for the two-dimensional discrete Laplacian.

    Args:
        non_zero_elements (list): A list of distinct non-zero elements [A0, A1, A2] in the two-dimensional discrete Laplacian.
            - A0: The diagonal element.
            - A1: The horizontal off-diagonal element.
            - A2: The vertical off-diagonal element.
        dim (int): The dimension of the two-dimensional discrete Laplacian, which must be a power of two.

    Returns:
        dict: The dictionary consisting of data items.
    """
    dim = nx * ny
    n = int(np.log2(dim))
    data_item = {
        0: [non_zero_elements[0],
            0,
            tools.binary_range(0, dim - 1, n, True, right_close=True)],
        1: [non_zero_elements[1],
            -1,
            tools.binary_range(0, dim - 1, n, True, True, 1)
            + tools.binary_range(0, dim - 1, n, True, True, 2)
            + tools.binary_range(0, dim - 1, n, True, True, 3)],
        2: [non_zero_elements[1],
            1,
            tools.binary_range(0, dim - 1, n, True, True, 0)
            + tools.binary_range(0, dim - 1, n, True, True, 1)
            + tools.binary_range(0, dim - 1, n, True, True, 2)],
        3: [non_zero_elements[2],
            -4,
            tools.binary_range(nx, dim - 1, n, True, True)],
        4: [non_zero_elements[2],
            4,
            tools.binary_range(0, dim - 1 - nx, n, True, True)]
    }

    return data_item


def test_two_dimensional_discrete_laplacian(data_item, dim):
    """
    Tests the construction of block encoding the two-dimensional discrete Laplacian.

    Args:
        data_item (dict): A dictionary representing the data item to be encoded.
            It should contain the coefficients and their corresponding binary ranges.
        dim (int): The dimension of the Laplacian matrix, which must be a power of two.

    Returns:
        tuple: A tuple containing the constructed quantum circuit and the encoded Laplacian matrix.
            - circuit: The quantum circuit of block encoding.
            - encoded_matrix: The encoded Laplacian matrix.
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
    delta_x = 1
    delta_y = 2
    nx = 4
    ny = 4
    parameters = {'delta_x': delta_x, 'delta_y': delta_y}

    # Dimension of matrix
    dim = nx * ny

    # Get all distinct non-zero elements
    non_zero_elements = get_non_zero_elements(parameters)

    # Get data items
    data_items = get_data_item(non_zero_elements, nx, ny)

    # Get the two-dimensional discrete Laplacian to be encoded
    matrix = two_dimensional_discrete_laplacian(non_zero_elements, nx, ny)

    # Get the circuit and the encoded two-dimensional discrete Laplacian
    circuit, encoded_matrix = test_two_dimensional_discrete_laplacian(data_items, dim)

    # Compute the subnormalization
    subnormalization = abs(non_zero_elements[0]) + 2 * (abs(non_zero_elements[1]) + abs(non_zero_elements[2]))

    print(circuit)
    matrix_pd = pd.DataFrame(matrix)
    matrix_pd.to_excel('Laplacian_nx_' + str(nx) + '_ny_' + str(ny) + '.xlsx')
    encoded_matrix_pd = pd.DataFrame(encoded_matrix)
    encoded_matrix_pd.to_excel('Laplacian_nx_' + str(nx) + '_ny_' + str(ny) + '_encoded.xlsx')
    print('Actual subnormalization:')
    print(np.linalg.norm(matrix) / np.linalg.norm(encoded_matrix))
    print('Theoretical subnormalization:')
    print(subnormalization)
    error = np.linalg.norm(matrix - subnormalization * encoded_matrix)
    print('The error of block encoding:')
    print(error)