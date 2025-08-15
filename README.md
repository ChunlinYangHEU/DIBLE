# DIBLE
Quantum circuits for implementing dictionary-based block encoding of sparse matrices

This algorithm is built on top of [mindquantum](https://www.mindspore.cn/mindquantum/docs/en/r0.6/index.html) in Python.

The paper titled "Dictionary-based Block Encoding of Sparse Matrices with Low Subnormalization and Circuit Depth" presents this work. It is published in the journal **Quantum** and can be accessed at [https://quantum-journal.org/papers/q-2025-07-22-1805/](https://quantum-journal.org/papers/q-2025-07-22-1805/).

## Install Python and Python packages

1. Download and install [Anaconda](https://www.anaconda.com/download)

2. Open the Anaconda Prompt
   
3. Create a virtual environment with Python 3.9.11 as an example

   ```
   conda create -n quantum python=3.9.11 -y
   conda activate quantum
   ```

3. Install Python packages

   ```
   pip install numpy
   ```
   ```
   pip install mindquantum
   ```
   Packages used to write matrix into Excel
   ```
   pip install pandas
   ```
   ```
   pip install openpyxl
   ```

## Note

Put the folder "[dible](https://github.com/ChunlinYangHEU/DISBLE/tree/main/dible)" under the root directory of your project
