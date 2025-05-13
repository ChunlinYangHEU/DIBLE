import numpy as np
import pandas as pd
from disble import blockencoding, tools, qgates
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


def oracle_oc():
    """
    Constructs the quantum circuit of oracle Oc for Hermitian block encoding.

    Returns:
         Circuit: circuit of oracle Oc.
    """
    # Initialize the circuit
    circuit_oc = Circuit()

    circuit_oc += X.on(4)

    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 2, 3, 6, 7],
                 control_states=[0, 0, 0, 0, 0])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 3],
                 control_states=[0, 0])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 2, 3, 8, 9],
                 control_states=[0, 0, 1, 0, 1])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 2, 3, 8],
                 control_states=[0, 0, 1, 1])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 2, 3, 8],
                 control_states=[0, 1, 1, 0])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 2, 3, 8, 9],
                 control_states=[0, 1, 1, 1, 0])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 2, 3, 6],
                 control_states=[1, 0, 0, 0])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[4],
                 control_qubits=[1, 2, 3, 6, 7],
                 control_states=[1, 0, 0, 1, 0])

    circuit_oc += SWAP([0, 1])
    circuit_oc += SWAP([1,2])
    qgates.qgate('X',
                 circuit_oc,
                 target_qubit=[2],
                 control_qubits=[1,3],
                 control_states=[0,1])
    circuit_oc += qgates.right_shift(target_qubits=[0, 1])
    circuit_oc += qgates.left_shift(target_qubits=[0, 1, 2, 3],
                                    control_qubits=[9],
                                    control_states=[1])
    circuit_oc += qgates.left_shift(target_qubits=[0, 1, 2],
                                    control_qubits=[8],
                                    control_states=[1])
    circuit_oc += qgates.left_shift(target_qubits=[0, 1],
                                    control_qubits=[7],
                                    control_states=[1])
    circuit_oc += qgates.left_shift(target_qubits=[0],
                                    control_qubits=[6],
                                    control_states=[1])

    return circuit_oc


def test_two_dimensional_discrete_laplacian_hermitian(data_value, dim):
    """
    Tests the construction of Hermitian block encoding the two-dimensional discrete Laplacian.

    Args:
        data_value (np.array): An array of data values.
        dim (int): The dimension of the Laplacian matrix, which must be a power of two.

    Returns:
        tuple: A tuple containing the constructed quantum circuit and the encoded Laplacian matrix.
            - circuit: The quantum circuit of Hermitian block encoding.
            - encoded_matrix: The encoded Laplacian matrix.
    """
    # Get the number of qubits in the circuit
    num_idx_qubits0 = tools.num_qubits(len(data_value))
    num_working_qubits = int(np.log2(dim))
    num_qubits = 2 * num_working_qubits + 2

    # Initialize the circuit
    circuit = Circuit()

    # PREP
    circuit_prep = blockencoding.oracle_prep(data_value)
    circuit_prep = apply(circuit_prep, list(range(num_working_qubits - num_idx_qubits0, num_working_qubits)))
    circuit += circuit_prep

    # Oc
    circuit_oc = oracle_oc()
    circuit += circuit_oc

    # SWAP
    circuit += SWAP.on([num_working_qubits, num_working_qubits + 1])
    for i in range(num_working_qubits):
        circuit += SWAP.on([i, i + num_working_qubits + 2])

    # Oc^{\dagger}
    circuit_oc_dagger = dagger(circuit_oc)
    circuit += circuit_oc_dagger

    # PREP^{\dagger}
    circuit += dagger(circuit_prep)

    # Get the unitary of circuit
    unitary = circuit.matrix()
    unitary = np.array(unitary).reshape(2 ** num_qubits, 2 ** num_qubits)

    # Get the encoded matrix
    encoded_matrix = blockencoding.get_encoded_matrix(unitary, num_working_qubits)

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

    # Get the two-dimensional discrete Laplacian to be encoded
    matrix = two_dimensional_discrete_laplacian(non_zero_elements, nx, ny)

    # Get the data value
    data_value = np.array([non_zero_elements[2], non_zero_elements[1], non_zero_elements[0], non_zero_elements[1], non_zero_elements[2]])

    # Get the circuit and the encoded two-dimensional discrete Laplacian
    circuit, encoded_matrix = test_two_dimensional_discrete_laplacian_hermitian(data_value, dim)

    # Compute the subnormalization
    subnormalization = abs(non_zero_elements[0]) + 2 * (abs(non_zero_elements[1]) + abs(non_zero_elements[2]))

    print(circuit)
    print('The two-dimensional discrete Laplacian to be encoded (A):')
    print(matrix)
    matrix_pd = pd.DataFrame(matrix)
    matrix_pd.to_excel('Laplacian_nx_' + str(nx) + '_ny_' + str(ny) + '.xlsx')
    print('The Hermitian encoded two-dimensional discrete Laplacian (A1):')
    print(encoded_matrix)
    encoded_matrix_pd = pd.DataFrame(encoded_matrix)
    encoded_matrix_pd.to_excel('Laplacian_nx_' + str(nx) + '_ny_' + str(ny) + '_Hermitian_encoded.xlsx')
    print('Actual subnormalization:')
    print(np.linalg.norm(matrix) / np.linalg.norm(encoded_matrix))
    print('Theoretical subnormalization:')
    print(subnormalization)
