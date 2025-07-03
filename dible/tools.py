import copy
import numpy as np


def polar_sqrt(value):
    """
    Computes the polar square root of a given value or array of values.

    Args:
        value (numpy.array or list or int or float or complex): The input value or array of values.
            The input can be a numpy array, a list, or a single numeric value (int, float, complex).

    Returns:
        numpy.array, list, or single value: The polar square root of the input value(s).
            The return type matches the input type.

    Raises:
        ValueError: If the input type is not supported.
    """
    if isinstance(value, np.ndarray):
        sqrt_value = np.array([])
        for c in value:
            norm = np.abs(c)
            phase = np.angle(c)
            sqrt_c = np.sqrt(norm) * np.exp(1j * phase / 2)
            sqrt_value = np.append(sqrt_value, sqrt_c)

        return sqrt_value
    elif isinstance(value, list):
        sqrt_value = []
        for c in value:
            norm = np.abs(c)
            phase = np.angle(c)
            sqrt_c = np.sqrt(norm) * np.exp(1j * phase / 2)
            sqrt_value.append(sqrt_c)

        return sqrt_value
    elif isinstance(value, (int, float, complex)):
        norm = np.abs(value)
        phase = np.angle(value)
        sqrt_value = np.sqrt(norm) * np.exp(1j * phase / 2)

        return sqrt_value
    else:
        raise ValueError('No supported type')


def num_qubits(length):
    """
    Calculates the number of qubits required to represent a given length of data.

    Args:
        length (int): The length of the data to be represented by qubits.
            It should be a positive integer.

    Returns:
        int: The number of qubits required to represent the given length.
    """
    if length == 1:
        n = 1
    else:
        n = int(np.ceil(np.log2(length)))

    return n


def gray_code(x):
    """
    Get the Gray code of the input number.

    Args:
        x (int): A non-negative integer.

    Returns:
        int: Gray code of the input number.
    """
    return x ^ (x >> 1)


def different_gray_codes_index(number1, number2, length):
    """
    Get the index of the different bit between the gray codes of number1 and number2

    Args:
        number1 (int): A non-negative integer.
        number2 (int): A non-negative integer.
        length (int): The length of the gray codes.

    Returns:
        int: The index of the different bit between the gray codes of number1 and number2.
    """
    gray_code_number1 = gray_code(number1)
    gray_code_number2 = gray_code(number2)
    diff = gray_code_number1 ^ gray_code_number2    # Calculate XOR of the two Gray codes
    index = length - 1 - int(np.log2(diff))

    return index


def phase_angle_matrix_inverse(N):
    """
    Generate a matrix which computes phase_angles from phase

    This function constructs a matrix that can be used to calculate phase angles from phase values.
    The matrix is built recursively by combining smaller matrices.

    Args:
        N (int): The dimension of the matrix. It must be a power of two.

    Returns:
        np.array: A NumPy array representing the matrix which computes phase angles.
    """
    matrix = [[-1, -1], [-1, 1]]
    if N != 2:
        k = 2
        while k != N:
            leftward = np.kron(matrix, [1/2, 1/2])
            rightward = np.kron(np.eye(k), [-1, 1])
            matrix = np.vstack([leftward, rightward])
            k = 2 * k

    return np.array(matrix)


def sfwht(input_array):
    """
    Computes the Scaled Fast Walsh-Hadamard Transform (SFWHT) of a given array.

    Args:
        input_array (numpy.array): The input array to be transformed. It can
            be a 1D column vector or a 2D square matrix with dimensions that are
            a power of 2.

    Returns:
        numpy.array: The transformed array after applying the SFWHT.
    """
    n = input_array.shape[0]
    k = int(np.log2(n))

    for h in range(1, k + 1):
        for i in range(1, n + 1, 2 ** h):
            for j in range(i, i + 2 ** (h - 1)):
                # Create a deep copy of the array to avoid modifying the original array
                temp = copy.deepcopy(input_array)
                # Extract the current element/row and the corresponding element/row to be combined
                x = temp[j - 1, :]
                y = temp[j + 2 ** (h - 1) - 1, :]
                # Update the array with the combined values according to the slant structure
                input_array[j - 1, :] = (x + y) / 2
                input_array[j + 2 ** (h - 1) - 1, :] = (x - y) / 2

    return input_array


def gray_permutation(input_array):
    """
    Reorders the rows of the input array according to the Gray code permutation.

    Args:
        input_array (numpy.array): The input array to be reordered. It can be
            a 1D column vector or a 2D matrix.

    Returns:
        numpy.array: The reordered array with rows permuted according to the
            Gray code sequence.
    """
    n = input_array.shape[0]
    new_array = np.zeros_like(input_array)

    for i in range(n):
        # Compute the Gray code for the current index
        gray_index = gray_code(i)
        # Reorder the rows based on the Gray code sequence
        new_array[i, :] = input_array[gray_index, :]

    return new_array


def binary_list(num, length=None):
    """
    Generate the binary list of the input number with the specified length.

    This function converts an integer into its binary representation as a list of integers.
    If the length parameter is provided, the binary representation will be padded with zeros to reach the specified length.

    Args:
        num (int): The number to convert to binary.
        length (int, optional): The desired length of the binary list. Defaults to None.

    Returns:
        list: A list of binary numbers (0s and 1s) representing the input number.
    """
    num_binary = bin(num)[2:]
    if length:
        num_binary = num_binary.zfill(length)

    return [int(bit) for bit in num_binary]


def __binary_range0(num, length, close=False, mod4_num=None):
    """
    Generate the binary representation of the range [0, num) (or [0, num] if 'close' is True) with a specified 'length'.

    This function takes an integer 'num' and returns a list of binary lists, each representing a range of numbers that,
        when combined with a C-NOT gate, will flip the target qubit if the control qubits are in the specified range.
    The 'mod4_num' parameter is used to filter the range further by requiring that
        the numbers in the range must also satisfy the condition of being congruent to 'mod4_num' modulo 4.

    Args:
        num (int): The upper limit of the range (exclusive if 'close' is False).
        length (int): The length of the binary representation.
        close (bool, optional): Whether the range includes 'num' itself. Defaults to False.
        mod4_num (int, optional): If provided, only include numbers in the range that are congruent to 'mod4_num' modulo 4.
                                   Defaults to None.

    Returns:
        list: A list of binary lists, each representing a range of numbers.

    Raises:
        ValueError: If 'num' is out of the valid range for the given 'length', or if 'num' is 0 and 'close' is False.
    """
    # Check if the input "num" is illegal
    if num > 2 ** length - 1 or num < 0 or (num == 0 and (not close)):
        raise ValueError('The input is illegal.')

    # Get the binary list of num
    num_binary = binary_list(num, length)

    # Create an empty list to save intervals
    range0_list = []

    # Get the interval [0, num)
    for index, bit in enumerate(num_binary):
        if bit == 1:
            range0 = num_binary[:index + 1]
            range0[-1] = 0
            if mod4_num is not None:
                if length < 2:
                    raise ValueError('')
                if index + 1 <= length - 2:
                    range0 += [None for _ in range(length - index - 3)] + binary_list(mod4_num, 2)
                    range0_list.append(range0)
                elif index + 1 == length - 1:
                    if mod4_num == 0 or mod4_num == 1:
                        range0.append(mod4_num)
                        range0_list.append(range0)
                elif index + 1 == length:
                    if (num_binary[-1] ^ 1) + num_binary[-2] * 2 == mod4_num:
                        range0_list.append(range0)
            else:
                range0_list.append(range0)

    # Add "num" into [0, num) if it is a closed interval
    if close:
        if mod4_num is None or ((mod4_num is not None) and (np.mod(num, 4) == mod4_num)):
            range0_list.append(num_binary)

    return range0_list


def binary_range(m, n, length, left_close=False, right_close=False, mod4_num=None):
    """
    Generate the binary representation of the range (m, n) with a specified 'length'.

    This function takes two integers 'm' and 'n' and returns a list of binary lists, each representing a range of numbers.
    The 'left_close' and 'right_close' parameters determine whether the range includes 'm' and 'n' respectively.
    The 'mod4_num' parameter is used to filter the range further by requiring that
        the numbers in the range must also satisfy the condition of being congruent to 'mod4_num' modulo 4.

    Args:
        m (int): The start of the range (exclusive unless 'left_close' is True).
        n (int): The end of the range (exclusive unless 'right_close' is True).
        length (int): The length of the binary representation.
        left_close (bool, optional): Whether the range includes 'm' itself. Defaults to False.
        right_close (bool, optional): Whether the range includes 'n' itself. Defaults to False.
        mod4_num (int, optional): If provided, only include numbers in the range that are congruent to 'mod4_num' modulo 4.
                                      Defaults to None.

    Returns:
        list: A list of binary lists, each representing a range of numbers.

    Raises:
        ValueError: If 'm' is greater than 'n', or if 'm' or 'n' is out of the valid range for the given 'length'.
    """
    # Check if the input numbers are legal
    if m < 0 or m > n:
        raise ValueError("Input number is illegal. m must be between 0 and n.")

    if m == 0:
        # if left_close and n + 1 == 2 ** length:
        #     return []
        if left_close:
            m_rang0_list = []
        else:
            m_rang0_list = [binary_list(0, length)]
    else:
        # Get the binary representation of range [0, m)
        m_rang0_list = __binary_range0(m, length, close=not left_close, mod4_num=mod4_num)

    # Get the binary representation of range [0, n]
    n_rang0_list = __binary_range0(n, length, close=right_close, mod4_num=mod4_num)

    # Get the binary representation of range [m, n]
    range_list = m_rang0_list + n_rang0_list

    return range_list


def reverse_index_bits(arr):
    """
    Reverse the binary index of the input array for all dimensions.

    This function takes an input array and returns a new array with the same values
    but with indices reversed in binary representation for each dimension.
    It works by recursively reversing the binary index for each dimension,
    starting from the last dimension and moving towards the first.

    Args:
        arr (np.ndarray): The input array with any number of dimensions.

    Returns:
        np.array: A new array with the same values as the input array but with reversed binary indices for all dimensions.

    Raises:
        ValueError: If the input is not a NumPy array.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    def _reverse_dim(arr, dim):
        """
        Reverse the binary index for a specific dimension of the array.

        Args:
            arr (np.ndarray): The input array.
            dim (int): The dimension to reverse the binary index for.

        Returns:
            np.ndarray: A new array with the binary index reversed for the specified dimension.
        """
        n = int(np.log2(arr.shape[dim]))
        new_arr = np.empty_like(arr)

        # Iterate over all indices in the current dimension
        for index in range(arr.shape[dim]):
            new_index_bin = '0b' + bin(index)[2:].zfill(n)[::-1]
            new_index = int(new_index_bin, 2)

            # Construct the full index tuple for the new and original arrays
            index_tuple = [slice(None)] * arr.ndim
            index_tuple[dim] = new_index
            index_tuple_original = [slice(None)] * arr.ndim
            index_tuple_original[dim] = index

            # Copy the values from the original array to the new array at the reversed index position
            new_arr[tuple(index_tuple)] = arr[tuple(index_tuple_original)]
        return new_arr

    # Start from the last dimension and move towards the first
    for dim in range(arr.ndim - 1, -1, -1):
        arr = _reverse_dim(arr, dim)

    return arr


def pad_to_power_of_two(arr):
    """
    Pads an array to the nearest power of two length.

    Args:
        arr (numpy.array or list): The input array to be padded.
            It should be a 1D array or list of numeric values.

    Returns:
        numpy.array: The padded array with a length that is a power of two.

    Raises:
        TypeError: If the input is not a numpy array or list.
    """
    length = len(arr)
    if length == 0:
        return arr
    if length == 1:
        new_arr = np.append(arr, 0)
        return new_arr
    n = np.log2(length)
    n_ceil = int(np.ceil(np.log2(length)))
    if n_ceil - n == 0:
        return arr
    else:
        pad_amount = 2 ** n_ceil - length

    return np.pad(arr, (0, pad_amount), mode='constant')