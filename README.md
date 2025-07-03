# DIBLE
Quantum circuits for implementing dictionary-based block encoding of sparse matrices

This algorithm is built on top of [mindquantum](https://www.mindspore.cn/mindquantum/docs/en/r0.6/index.html) in Python.

This work is presented in the paper titled 'Dictionary-based block encoding of sparse matrices with low subnormalization and circuit depth' available on arXiv at [https://doi.org/10.48550/arXiv.2405.18007](
https://doi.org/10.48550/arXiv.2405.18007).

## 1. Install Python and Python packages

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
