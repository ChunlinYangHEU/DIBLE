import numpy as np
import pandas as pd
import time
from mindquantum import *
from dsble import tools, blockencoding


def get_non_zero_elements(parameters):
    """
    Computes all distinct non-zero elements for matrices A and B according to given parameters.

    Args:
        parameters (dict): A dictionary containing the necessary parameters.
            - 'depth_ice': The depth of the ice layer.
            - 'num_ice': The number of grid points in the ice layer.
            - 'depth_seawater': The depth of the seawater layer.
            - 'num_seawater': The number of grid points in the seawater layer.
            - 'density': The density of the medium at depth z.
            - 'sound_velocity': The sound velocity in the medium at depth z.
            - 'frequency': The frequency of the sound wave.
            - 'mu_ice': The Lamé coefficient mu for ice.
            - 'lambda_ice': The Lamé coefficient lambda for ice.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - non_zero_element_A: All distinct non-zero elements of matrix A.
            - non_zero_element_B: All distinct non-zero elements of matrix B.
    """
    # Get parameters
    depth_ice = parameters['depth_ice']  # The depth of ice
    num_ice = parameters['num_ice']  # The number of grids
    delta_ice = depth_ice / num_ice  # The depth of each grid
    depth_seawater = parameters['depth_seawater']  # The depth of seawater
    num_seawater = parameters['num_seawater']  # The number of grids
    delta_seawater = depth_seawater / num_seawater  # The depth of each grid
    density = parameters['density']  # Density at depth z
    sound_velocity = parameters['sound_velocity']  # Sound velocity at depth z
    frequency = parameters['frequency']  # Frequency of sound wave
    angle_frequency = 2 * np.pi * frequency  # Angular frequency of sound wave
    mu_ice = parameters['mu_ice']  # Lamé coefficients of ice
    lambda_ice = parameters['lambda_ice']  # Lamé coefficients of ice

    # Related parameters
    x1 = 1 / mu_ice
    x2 = 1 / (lambda_ice + 2 * mu_ice)
    x3 = lambda_ice / (lambda_ice + 2 * mu_ice)
    x4 = (4 * mu_ice * (lambda_ice + mu_ice)) / (lambda_ice + 2 * mu_ice)
    x5 = - density * (angle_frequency ** 2)

    # Get all distinct non-zero elements in matrice A and B
    a0 = 2 / delta_ice
    a1 = x5
    a2 = -1
    a3 = 1
    a4 = x1
    a5 = x2
    a6 = -x3
    a7 = -2 / delta_ice
    a8 = -delta_ice * density * angle_frequency ** 2
    a9 = delta_seawater ** 2 * angle_frequency ** 2 / (2 * sound_velocity ** 2) - 1
    a10 = -2 + delta_seawater ** 2 * angle_frequency ** 2 / sound_velocity ** 2
    a11 = 1 / delta_seawater
    a12 = -1 / delta_seawater + delta_seawater * angle_frequency ** 2 / (2 * sound_velocity ** 2)
    b0 = - x3
    b1 = -1
    b2 = -x4
    b3 = delta_seawater ** 2 / 2
    b4 = delta_seawater ** 2
    b5 = delta_seawater / 2
    non_zero_element_A = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12])
    non_zero_element_B = np.array([b0, b1, b2, b3, b4, b5])

    return non_zero_element_A, non_zero_element_B


def matrix_AB(non_zero_elements_A, non_zero_elements_B, num_ice, num_seawater):
    """
    Constructs matrices A and B according to the given non-zero elements and grid dimensions.

    Args:
        non_zero_elements_A (numpy.ndarray): An array of non-zero elements for matrix A.
            These elements are derived from physical properties and discretization parameters.
        non_zero_elements_B (numpy.ndarray): An array of non-zero elements for matrix B.
            These elements are derived from physical properties and discretization parameters.
        num_ice (int): The number of grids in the ice layer.
        num_seawater (int): The number of grids in the seawater layer.

    Returns:
        tuple: A tuple containing the constructed matrices A and B.
            - matrix_A: The matrix A in the ocean acoustic GEPs.
            - matrix_B: The matrix B in the ocean acoustic GEPs.
    """
    # The dimension of matrices A and B
    dim = 4 * num_ice + num_seawater + 5
    matrix_A = np.zeros(shape=(dim, dim))
    matrix_B = np.zeros(shape=(dim, dim))

    # All non-zero elements in matrix A
    for n1 in range(num_ice):
        matrix_A[4 * n1 + 2, 4 * n1] = non_zero_elements_A[0]
        matrix_A[4 * n1 + 3, 4 * n1 + 1] = non_zero_elements_A[0]
        matrix_A[4 * n1 + 4, 4 * n1 + 2] = non_zero_elements_A[0]
        matrix_A[4 * n1 + 5, 4 * n1 + 3] = non_zero_elements_A[0]
        matrix_A[4 * n1 + 4, 4 * n1] = non_zero_elements_A[1]
        matrix_A[4 * n1 + 5, 4 * n1 + 1] = non_zero_elements_A[1]
        matrix_A[4 * n1 + 4, 4 * n1 + 4] = non_zero_elements_A[1]
        matrix_A[4 * n1 + 5, 4 * n1 + 5] = non_zero_elements_A[1]
        matrix_A[4 * n1 + 2, 4 * n1 + 1] = non_zero_elements_A[2]
        matrix_A[4 * n1 + 2, 4 * n1 + 5] = non_zero_elements_A[2]
        matrix_A[4 * n1 + 2, 4 * n1 + 2] = non_zero_elements_A[4]
        matrix_A[4 * n1 + 2, 4 * n1 + 6] = non_zero_elements_A[4]
        matrix_A[4 * n1 + 3, 4 * n1 + 3] = non_zero_elements_A[5]
        matrix_A[4 * n1 + 3, 4 * n1 + 7] = non_zero_elements_A[5]
        matrix_A[4 * n1 + 4, 4 * n1 + 3] = non_zero_elements_A[6]
        matrix_A[4 * n1 + 4, 4 * n1 + 7] = non_zero_elements_A[6]
        matrix_A[4 * n1 + 2, 4 * n1 + 4] = non_zero_elements_A[7]
        matrix_A[4 * n1 + 3, 4 * n1 + 5] = non_zero_elements_A[7]
        matrix_A[4 * n1 + 4, 4 * n1 + 6] = non_zero_elements_A[7]
        matrix_A[4 * n1 + 5, 4 * n1 + 7] = non_zero_elements_A[7]
    matrix_A[0, 2] = non_zero_elements_A[3]
    matrix_A[1, 3] = non_zero_elements_A[3]
    matrix_A[4 * num_ice + 2, 4 * num_ice + 2] = non_zero_elements_A[3]
    matrix_A[4 * num_ice + 3, 4 * num_ice + 3] = non_zero_elements_A[3]
    matrix_A[4 * num_ice + 3, 4 * num_ice + 4] = non_zero_elements_A[3]
    matrix_A[4 * num_ice + 4, 4 * num_ice + 5] = non_zero_elements_A[3]
    for n2 in range(num_seawater - 1):
        matrix_A[4 * num_ice + 5 + n2, 4 * num_ice + 4 + n2] = non_zero_elements_A[3]
        matrix_A[4 * num_ice + 5 + n2, 4 * num_ice + 6 + n2] = non_zero_elements_A[3]
        matrix_A[4 * num_ice + 5 + n2, 4 * num_ice + 5 + n2] = non_zero_elements_A[10]
    matrix_A[4 * num_ice + 4, 4 * num_ice + 1] = non_zero_elements_A[8]
    matrix_A[4 * num_ice + 4, 4 * num_ice + 4] = non_zero_elements_A[9]
    matrix_A[4 * num_ice + num_seawater + 4, 4 * num_ice + num_seawater + 3] = non_zero_elements_A[11]
    matrix_A[4 * num_ice + num_seawater + 4, 4 * num_ice + num_seawater + 4] = non_zero_elements_A[12]

    # All non-zero elements in matrix B
    for n1 in range(num_ice):
        matrix_B[4 * n1 + 3, 4 * n1] = non_zero_elements_B[0]
        matrix_B[4 * n1 + 3, 4 * n1 + 4] = non_zero_elements_B[0]
        matrix_B[4 * n1 + 5, 4 * n1 + 2] = non_zero_elements_B[1]
        matrix_B[4 * n1 + 5, 4 * n1 + 6] = non_zero_elements_B[1]
        matrix_B[4 * n1 + 4, 4 * n1] = non_zero_elements_B[2]
        matrix_B[4 * n1 + 4, 4 * n1 + 4] = non_zero_elements_B[2]
    for n2 in range(num_seawater - 1):
        matrix_B[4 * num_ice + 5 + n2, 4 * num_ice + 5 + n2] = non_zero_elements_B[4]
    matrix_B[4 * num_ice + 4, 4 * num_ice + 4] = non_zero_elements_B[3]
    matrix_B[4 * num_ice + num_seawater + 4, 4 * num_ice + num_seawater + 4] = non_zero_elements_B[5]

    return matrix_A, matrix_B


def get_data_item(non_zero_elements_A, non_zero_elements_B, num_ice, num_seawater):
    """
    Obtains the data items for block encoding of matrices A and B.

    Args:
        non_zero_elements_A (numpy.array): An array of all distinct non-zero elements for matrix A.
            These elements are derived from physical properties and discretization parameters.
        non_zero_elements_B (numpy.array): An array of all distinct non-zero elements for matrix B.
            These elements are derived from physical properties and discretization parameters.
        num_ice (int): The number of grids in the ice layer.
        num_seawater (int): The number of grids in the seawater layer.

    Returns:
        tuple: A tuple containing two dictionaries:
            - data_item_A: The dictionary consisting of data items for matrix A.
            - data_item_B: The dictionary consisting of data items for matrix B.
    """
    # Dimension of the matrix
    dim = 4 * num_ice + num_seawater + 5
    n = int(np.ceil(np.log2(dim)))

    # The data items of matrix A
    UA5 = np.zeros(shape=(dim, dim))
    UA5[2, 0] = 1
    UA5[3, 1] = 1
    UA5[0, 2] = 1
    UA5[1, 3] = 1
    for j in range(4, 4 * num_ice + 4):
        UA5[j, j] = 1
    for j in range(4 * num_ice + 4, 4 * num_ice + num_seawater + 4):
        UA5[j + 1, j] = 1
    UA5[4 * num_ice + 4, 4 * num_ice + num_seawater + 4] = 1
    UA5 = tools.reverse_index_bits(UA5)
    data_item_A = {
        0: [non_zero_elements_A[0],
            2,
            tools.binary_range(0, 4 * num_ice - 1, n, True, True)],
        1: [non_zero_elements_A[1],
            4,
            tools.binary_range(0, 4 * num_ice - 3, n, True, True, 0)
            + tools.binary_range(0, 4 * num_ice - 3, n, True, True, 1)],
        2: [non_zero_elements_A[1],
            0,
            tools.binary_range(4, 4 * num_ice + 1, n, True, True, 0)
            + tools.binary_range(4, 4 * num_ice + 1, n, True, True, 1)],
        3: [non_zero_elements_A[2],
            1,
            tools.binary_range(1, 4 * num_ice - 3, n, True, True, 1)],
        4: [non_zero_elements_A[2],
            -3,
            tools.binary_range(5, 4 * num_ice + 1, n, True, True, 1)],
        5: [non_zero_elements_A[3],
            UA5,
            [tools.binary_list(2, n), tools.binary_list(3, n),
             tools.binary_list(4 * num_ice + 2, n), tools.binary_list(4 * num_ice + 3, n)]
            + tools.binary_range(4 * num_ice + 4, 4 * num_ice + num_seawater + 2, n, True, True)],
        6: [non_zero_elements_A[3],
            -1,
            tools.binary_range(4 * num_ice + 4, 4 * num_ice + num_seawater + 4, n, True, True)],
        7: [non_zero_elements_A[4],
            0,
            tools.binary_range(2, 4 * num_ice - 2, n, True, True, 2)],
        8: [non_zero_elements_A[4],
            -4,
            tools.binary_range(6, 4 * num_ice + 2, n, True, True, 2)],
        9: [non_zero_elements_A[5],
            -4,
            tools.binary_range(7, 4 * num_ice + 3, n, True, True, 3)],
        10: [non_zero_elements_A[5],
             0,
             tools.binary_range(3, 4 * num_ice - 1, n, True, True, 3)],
        11: [non_zero_elements_A[6],
             -3,
             tools.binary_range(7, 4 * num_ice + 3, n, True, True, 3)],
        12: [non_zero_elements_A[6],
             1,
             tools.binary_range(3, 4 * num_ice - 1, n, True, True, 3)],
        13: [non_zero_elements_A[7],
             -2,
             tools.binary_range(4, 4 * num_ice + 3, n, True, True)],
        14: [non_zero_elements_A[8],
             3,
             [tools.binary_list(4 * num_ice + 1, n)]],
        15: [non_zero_elements_A[9],
             0,
             [tools.binary_list(4 * num_ice + 4, n)]],
        16: [non_zero_elements_A[10],
             0,
             tools.binary_range(4 * num_ice + 5, 4 * num_ice + num_seawater + 3, n, True, True)],
        17: [non_zero_elements_A[11],
             1,
             [tools.binary_list(4 * num_ice + num_seawater + 3, n)]],
        18: [non_zero_elements_A[12],
             0,
             [tools.binary_list(4 * num_ice + num_seawater + 4, n)]]
    }

    # The data items of matrix B
    data_item_B = {
        0: [non_zero_elements_B[0],
            3,
            tools.binary_range(0, 4 * num_ice - 4, n, True, True, 0)],
        1: [non_zero_elements_B[0],
            -1,
            tools.binary_range(4, 4 * num_ice, n, True, True, 0)],
        2: [non_zero_elements_B[1],
            3,
            tools.binary_range(2, 4 * num_ice - 2, n, True, True, 2)],
        3: [non_zero_elements_B[1],
            -1,
            tools.binary_range(6, 4 * num_ice + 2, n, True, True, 2)],
        4: [non_zero_elements_B[2],
            4,
            tools.binary_range(0, 4 * (num_ice - 1), n, True, True, 0)],
        5: [non_zero_elements_B[2],
            0,
            tools.binary_range(4, 4 * num_ice, n, True, True, 0)],
        6: [non_zero_elements_B[3],
            0,
            [tools.binary_list(4 * num_ice + 4, n)]],
        7: [non_zero_elements_B[4],
            0,
            tools.binary_range(4 * num_ice + 5, 4 * num_ice + num_seawater + 3, n, True, True)],
        8: [non_zero_elements_B[5],
            0,
            [tools.binary_list(4 * num_ice + num_seawater + 4, n)]]
    }

    return data_item_A, data_item_B


def test_ocean_acoustic_GEPs(data_item, dim):
    """
    Tests the block encoding of matrices related to generalized eigenvalue problems in ocean acoustics.

    Args:
        data_item (dict): The dictionary consisting of data items.
        dim (int): The dimension of the matrices, which must be a power of two.

    Returns:
        tuple: A tuple containing the constructed quantum circuit and the encoded matrix.
            - circuit: The quantum circuit of block encoding.
            - encoded_matrix: The encoded matrix.
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
    # Set parameters
    # Note: 4 * num_ice + num_seawater + 5 should be a power of two.
    depth_ice = 5  # The depth of ice
    num_ice = 2  # The number of grids of ice
    depth_seawater = 7  # The depth of seawater
    num_seawater = 3  # The number of grids of seawater
    density = 1.1  # Density at depth z, assumed to be constant
    sound_velocity = 10  # Sound velocity at depth z, assumed to be constant
    frequency = 5  # Frequency of sound wave
    lambda_ice = 5  # Lamé coefficients of ice at depth z, assumed to be constant
    mu_ice = 5  # Lamé coefficients of ice at depth z, assumed to be constant
    parameters = {
        'depth_ice': depth_ice,
        'num_ice': num_ice,
        'depth_seawater': depth_seawater,
        'num_seawater': num_seawater,
        'density': density,
        'sound_velocity': sound_velocity,
        'frequency': frequency,
        'lambda_ice': lambda_ice,
        'mu_ice': mu_ice
    }

    # The dimension of matrices A and B
    dim = 4 * num_ice + num_seawater + 5

    # Get all distinct non-zero elements in matrice A and B
    non_zero_elements_A, non_zero_elements_B = get_non_zero_elements(parameters)

    # Get the signless Laplacian matrix to be encoded
    matrix_A, matrix_B = matrix_AB(non_zero_elements_A, non_zero_elements_B, num_ice, num_seawater)

    start_time = time.time()

    # Get data items of matrices A and B
    data_item_A, data_item_B = get_data_item(non_zero_elements_A, non_zero_elements_B, num_ice, num_seawater)

    # Get the circuit and the encoded matrix A
    circuit_A, encoded_matrix_A = test_ocean_acoustic_GEPs(data_item_A, dim=dim)
    circuit_B, encoded_matrix_B = test_ocean_acoustic_GEPs(data_item_B, dim=dim)

    # Compute the subnormalization of matrices A and B
    subnormalization_A = (abs(non_zero_elements_A[0]) + 2 * (np.sum(np.abs(non_zero_elements_A[1:7])))
                          + np.sum(np.abs(non_zero_elements_A[7:])))
    subnormalization_B = 2 * np.sum(np.abs(non_zero_elements_B[0:3])) + np.sum(np.abs(non_zero_elements_B[3:]))

    print(circuit_A)
    matrix_A_pd = pd.DataFrame(matrix_A)
    matrix_A_pd.to_excel('matrix_A.xlsx')
    encoded_matrix_A_pd = pd.DataFrame(encoded_matrix_A)
    encoded_matrix_A_pd.to_excel('matrix_A_encoded.xlsx')
    print('Actual subnormalization of A:')
    print(np.linalg.norm(matrix_A) / np.linalg.norm(encoded_matrix_A))
    print('Theoretical subnormalization of A:')
    print(subnormalization_A)
    error_A = np.linalg.norm(matrix_A - subnormalization_A * encoded_matrix_A)
    print('The error of block encoding:')
    print(error_A)
    print('==============================================================================================')

    print(circuit_B)
    matrix_B_pd = pd.DataFrame(matrix_B)
    matrix_B_pd.to_excel('matrix_B.xlsx')
    encoded_matrix_B_pd = pd.DataFrame(encoded_matrix_B)
    encoded_matrix_B_pd.to_excel('matrix_B_encoded.xlsx')
    print('Actual subnormalization of B:')
    print(np.linalg.norm(matrix_B) / np.linalg.norm(encoded_matrix_B))
    print('Theoretical subnormalization of B:')
    print(subnormalization_B)
    error_B = np.linalg.norm(matrix_B - subnormalization_B * encoded_matrix_B)
    print('The error of block encoding:')
    print(error_B)

    print('==============================================================================================')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time：{elapsed_time} seconds")
