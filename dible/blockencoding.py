"""
Dictionary-based sparse block encoding:
        idx —\——|PREP|————⊘————|UNPREP|——
                          |
        del ———————————|    |————————————
                       | Oc |
    \ket{j} —\—————————|    |———————————— \ket{i}
"""
import numpy as np
from mindquantum import *
from dible import tools, qgates, anglecompute


def qcircuit(data_item, num_working_qubits, control_qubits=None, control_states=None):
    """
    Constructs the quantum circuit for block encoding a sparse structured matrix with dictionary.

    Args:
        data_item (dict): Dictionary of sparse matrix, {l: [Al, cl(j), Sc(l)]}.
        num_working_qubits (int): Number of working qubits.
        control_qubits (list or int): List of control qubits.
        control_states (list or int): List of control states.

    Returns:
        Circuit: The quantum circuit of block encoding a sparse structured matrix.
    """
    # Get data value, data function and column index set from the dictionary
    data_value = np.array([])
    data_function = []
    column_index_set = []
    for l in range(len(data_item)):
        data_value = np.append(data_value, data_item[l][0])
        data_function.append(data_item[l][1])
        column_index_set.append(data_item[l][2])
    data_value = data_value.astype(complex)

    # Get the number of qubits in the circuit
    num_idx_qubits = tools.num_qubits(len(data_item))

    # Initialize the circuit
    circuit = Circuit()

    # Get the circuit of oracles
    circuit_prep = oracle_prep(data_value)
    circuit_oc = oracle_oc(data_function, column_index_set, num_idx_qubits, num_working_qubits)
    circuit_unprep = oracle_prep(np.conj(data_value))
    circuit_unprep = dagger(circuit_unprep)

    circuit += circuit_prep
    circuit += circuit_oc
    circuit += circuit_unprep

    return circuit


def oracle_prep(data_value):
    """
    Construct the quantum circuit of oracle PREP according to data value.

    Args:
        data_value (np.ndarray): Data values of all data items.

    Returns:
        Circuit: The quantum circuit of oracle PREP.
    """
    # Initialize the circuits
    circuit_prep = Circuit()

    # Turn the data_values into a quantum state
    subnormalization = np.sum(np.abs(data_value))
    data_value = tools.pad_to_power_of_two(data_value)
    state = tools.polar_sqrt(data_value) / np.sqrt(subnormalization)
    num_qubits = int(np.log2(len(state)))

    # Determine whether the state is complex or real
    if np.issubdtype(state.dtype, np.complexfloating):
        is_real = False
    elif np.issubdtype(state.dtype, np.floating) or np.issubdtype(state.dtype, np.integer):
        is_real = True
    else:
        raise ValueError('Unsupported data type')

    # Construct the circuit
    if is_real:
        norm_angles = anglecompute.binarytree_vector(state, 'norm', True)
        qgates.qgate('RY',
                     circuit_prep,
                     target_qubit=[0],
                     rotation_angle=norm_angles[0])
        angle_index = 1
        for layer in range(1, num_qubits):
            ur_angles = anglecompute.uniformly_rotation_angles(norm_angles[angle_index: angle_index + 2 ** layer])
            circuit_ur = qgates.compress_uniformly_rotation('RY',
                                                            target_qubit=layer,
                                                            control_qubits=list(range(layer)),
                                                            rotation_angles=ur_angles)
            circuit_prep += circuit_ur
            angle_index += 2 ** layer
    else:
        norm = np.abs(state)
        phase = np.angle(state)
        norm_angles = anglecompute.binarytree_vector(norm, 'norm')
        phase_angles = anglecompute.binarytree_vector(phase, 'phase')

        qgates.qgate('RZ',
                     circuit_prep,
                     target_qubit=[0],
                     rotation_angle=phase_angles[0])
        qgates.qgate('RY',
                     circuit_prep,
                     target_qubit=[0],
                     rotation_angle=norm_angles[0])
        angle_index = 1
        for layer in range(1, num_qubits):
            ur_angles = anglecompute.uniformly_rotation_angles(norm_angles[angle_index: angle_index + 2 ** layer])
            circuit_ur = qgates.compress_uniformly_rotation('RY',
                                                            target_qubit=layer,
                                                            control_qubits=list(range(layer)),
                                                            rotation_angles=ur_angles)
            circuit_prep += circuit_ur
            angle_index += 2 ** layer

        qgates.qgate('RZ',
                     circuit_prep,
                     target_qubit=0,
                     rotation_angle=phase_angles[1])
        angle_index = 2
        for layer in range(1, num_qubits):
            ur_angles = anglecompute.uniformly_rotation_angles(phase_angles[angle_index: angle_index + 2 ** layer])
            circuit_ur = qgates.compress_uniformly_rotation('RZ',
                                                            target_qubit=layer,
                                                            control_qubits=list(range(layer)),
                                                            rotation_angles=ur_angles)
            circuit_prep += circuit_ur
            angle_index += 2 ** layer

    return circuit_prep


def oracle_oc(data_function, column_index_set, num_idx_qubits, num_working_qubits):
    """
    Constructs the quantum circuit for oracle Oc according to the data function and column_index_set.

    Args:
        data_function (list[int or np.array]): List of data functions of all data items.
        column_index_set (list[list[list[int]]]): List of column indices of all data items.
        num_idx_qubits (int): Number of qubits of register idx.
        num_working_qubits (int): Number of working qubits.

    Returns:
        Circuit: The quantum circuit of oracle Oc.
    """
    # Initialize the circuit
    circuit_oc = Circuit()

    for i in range(num_working_qubits):
        circuit_oc += I.on(num_idx_qubits + 1 + i)

    # X
    circuit_oc += X.on(num_idx_qubits)

    # Oc1
    # Create two lists to save control qubits and control states of MC-NOT gates
    ctrl_qubits_list = []
    ctrl_states_list = []

    # Get MC-NOT gates according to column_index_set
    for data_index, scl in enumerate(column_index_set):
        for qbits_states in scl:
            qbits = list(range(num_idx_qubits))
            states = tools.binary_list(data_index, num_idx_qubits)
            for qbit, state in enumerate(qbits_states):
                if state is not None:
                    qbits.append(num_idx_qubits + 1 + qbit)
                    states.append(state)
            ctrl_qubits_list.append(qbits)
            ctrl_states_list.append(states)

    # Simplify MC-NOT gates
    ctrl_qubits_list, ctrl_states_list = qgates.mcnots_simplify(ctrl_qubits_list, ctrl_states_list)

    # Perform MC-NOT gates
    for (ctrl_qubits, ctrl_states) in zip(ctrl_qubits_list, ctrl_states_list):
        qgates.qgate('X',
                     circuit_oc,
                     target_qubit=num_idx_qubits,
                     control_qubits=ctrl_qubits,
                     control_states=ctrl_states)

    # Oc2, Oc3
    if not all((isinstance(x, int) and x == 0) or (isinstance(x, np.ndarray) and np.array_equal(x, np.eye(x.shape[0])))
               for x in data_function):
        # Create empty lists to save the left-shift and right-shift gates, custom unitaries
        lk_target_qubits_list = []
        lk_control_qubits_list = []
        lk_control_states_list = []
        rk_target_qubits_list = []
        rk_control_qubits_list = []
        rk_control_states_list = []
        custom_unitary_list = []
        custom_unitary_control_qubits_list = []
        custom_unitary_control_states_list = []

        # Get data function and construct its circuit
        # data function: (1) an integer k, which refers to the function j+k
        #                (2) custom n-qubit unitary
        for data_index, func in enumerate(data_function):
            # data function (1)
            if isinstance(func, int):
                if func != 0:
                    func_binary = bin(np.abs(func))[2:].zfill(num_working_qubits)
                    for bit_index, bit in enumerate(func_binary):
                        if bit == '1':
                            trgt_qubits = list(range(num_idx_qubits+1, num_idx_qubits+1+bit_index + 1))
                            ctrl_qubits = list(range(num_idx_qubits))
                            ctrl_states = tools.binary_list(data_index, num_idx_qubits)
                            if func > 0:
                                lk_target_qubits_list.append(trgt_qubits)
                                lk_control_qubits_list.append(ctrl_qubits)
                                lk_control_states_list.append(ctrl_states)
                            elif func < 0:
                                rk_target_qubits_list.append(trgt_qubits)
                                rk_control_qubits_list.append(ctrl_qubits)
                                rk_control_states_list.append(ctrl_states)
            # data function (2)
            elif isinstance(func, np.ndarray):
                ctrl_qubits = list(range(num_idx_qubits))
                ctrl_states = tools.binary_list(data_index, num_idx_qubits)
                custom_unitary_list.append(func)
                custom_unitary_control_qubits_list.append(ctrl_qubits)
                custom_unitary_control_states_list.append(ctrl_states)
            else:
                raise ValueError('Unsupported data function type')

        # Simplify left and right shift gates
        (lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list,
         rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list) \
            = qgates.shift_gates_simplify1(
            lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list,
            rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list)

        # Perform left-shift and right-shift gates
        for lk_index, lk_trgt_qubits in enumerate(lk_target_qubits_list):
            circuit_oc += qgates.left_shift(target_qubits=lk_trgt_qubits,
                                            control_qubits=lk_control_qubits_list[lk_index],
                                            control_states=lk_control_states_list[lk_index])
        for rk_index, rk_trgt_qubits in enumerate(rk_target_qubits_list):
            circuit_oc += qgates.right_shift(target_qubits=rk_trgt_qubits,
                                             control_qubits=rk_control_qubits_list[rk_index],
                                             control_states=rk_control_states_list[rk_index])

        # Perform custom unitary
        for unitary_index, unitary in enumerate(custom_unitary_list):
            qgates.qgate(unitary,
                         circuit=circuit_oc,
                         target_qubit=list(range(num_idx_qubits+1, num_idx_qubits+1+num_working_qubits)),
                         control_qubits=custom_unitary_control_qubits_list[unitary_index],
                         control_states=custom_unitary_control_states_list[unitary_index],
                         gate_name='U'+str(unitary_index))

    return circuit_oc


def get_encoded_matrix(circuit, num_qubits, num_working_qubits):
    """
    Obtain the block-encoded matrix from the given block-encoding circuit.

    Args:
        circuit (Circuit): the block-encoding circuit.
        num_qubits (int): the number of qubits.
        num_working_qubits (int): The number of working qubits used in the encoding.

    Returns:
        numpy.array: The block-encoded matrix.
    """
    unitary = circuit.matrix()
    unitary = np.array(unitary).reshape(2 ** num_qubits, 2 ** num_qubits)
    unitary = tools.reverse_index_bits(unitary)
    matrix = unitary[: 2 ** num_working_qubits, : 2 ** num_working_qubits]

    return matrix
