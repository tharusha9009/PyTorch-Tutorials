# PyTorch-Tutorials
PyTorch tutorials 

## Pytorch_Fundemenentals.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tharusha9009/PyTorch-Tutorials/blob/main/Pytorch_Fundemenentals.ipynb)

This Jupyter notebook (`Pytorch_Fundemenentals.ipynb`) serves as a comprehensive introduction to the fundamental concepts of PyTorch, with a primary focus on **tensors**. Tensors are multi-dimensional arrays that are the core data structure in PyTorch, similar to NumPy arrays but with the added benefit of GPU acceleration, which is crucial for deep learning.

The notebook walks through the following key areas:

### 1. Introduction to Tensors
- **What are Tensors?** Introduction to scalars (0D), vectors (1D), matrices (2D), and higher-dimensional tensors.
- **Creating Tensors:**
    - Using `torch.tensor()` to create tensors from Python lists or numerical values.
    - Understanding tensor attributes:
        - `.ndim`: Number of dimensions.
        - `.shape`: Size of each dimension.
        - `.item()`: Get the Python number from a scalar tensor.

### 2. Random Tensors
- **Why Random Tensors?** Random tensors are crucial in neural networks because:
  - Neural networks start with tensors full of random numbers
  - Through training, these random numbers are adjusted to better represent the data
  - The learning process follows: `start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers`
- **Specialized Tensor Creation:**
    - `torch.rand(size)`: Tensors with random numbers uniformly distributed between 0 and 1.
    - `torch.zeros(size)`: Tensors filled with zeros.
    - `torch.ones(size)`: Tensors filled with ones.
    - `torch.arange(start, end, step)`: Tensors with a sequence of numbers.
    - `torch.zeros_like(input_tensor)`, `torch.ones_like(input_tensor)`: Tensors with the same shape as an existing tensor, filled with zeros or ones.

### 3. Tensor Datatypes
- Importance of using the correct datatype for tensors (e.g., `torch.float32`, `torch.float16`, `torch.int64`).
- Common issues arising from mismatched datatypes in operations.
- **The 3 Big Errors in PyTorch & Deep Learning:**
  1. Tensors not right datatype
  2. Tensors not right shape  
  3. Tensors not on the right device
- Specifying `dtype` during tensor creation (e.g., `torch.tensor([1, 2, 3], dtype=torch.float32)`).
- Changing tensor datatype using the `.type(new_dtype)` method.
- **Getting Information from Tensors (Tensor Attributes):**
  - Checking tensor datatype with `tensor.dtype`.
  - Checking tensor shape with `tensor.shape`.
  - Checking the device a tensor is on with `tensor.device` (e.g., 'cpu', 'cuda').

### 4. Tensor Manipulations and Operations
- **Basic Arithmetic:**
    - Element-wise addition, subtraction, multiplication, and division.
    - Using both overloaded operators (e.g., `tensor + 10`) and PyTorch functions (e.g., `torch.add(tensor, 10)`).
- **Matrix Multiplication:**
    - Distinction between element-wise multiplication (`*`) and matrix multiplication (`torch.matmul()` or `@` or `torch.mm()`).
    - **Rules for matrix multiplication:**
        - Inner dimensions must match (e.g., `(a, b) @ (b, c)`).
        - The resulting matrix shape is determined by the outer dimensions (e.g., `(a, c)`).
    - Using `.T` or `tensor.transpose()` to transpose matrices to satisfy shape requirements for multiplication.
    - **Common shape errors in deep learning** and how to resolve them using transpose operations.

### 5. Tensor Aggregations
- **Finding statistics:**
    - Finding minimum (`torch.min(tensor)` or `tensor.min()`).
    - Finding maximum (`torch.max(tensor)` or `tensor.max()`).
    - Calculating the mean (`torch.mean(tensor.type(torch.float32))` or `tensor.type(torch.float32).mean()`). Note: `mean()` typically requires floating-point input.
    - Summing all elements (`torch.sum(tensor)` or `tensor.sum()`).
- **Positional Min/Max:**
    - Finding the index of the minimum value (`tensor.argmin()`).
    - Finding the index of the maximum value (`tensor.argmax()`).

### 6. Reshaping, Viewing, Stacking, Squeezing, and Unsqueezing
- **Reshaping (`tensor.reshape(new_shape)`):** Changes the shape of a tensor. The total number of elements must remain the same.
- **Viewing (`tensor.view(new_shape)`):** Returns a new tensor with the desired shape that shares the same underlying data (memory) as the original tensor. Modifying the view will modify the original tensor.
- **Stacking (`torch.stack(list_of_tensors, dim)`):** Concatenates a sequence of tensors along a new dimension.
- **Squeezing (`tensor.squeeze()`):** Removes all dimensions of size 1 from a tensor.
- **Unsqueezing (`tensor.unsqueeze(dim)`):** Adds a dimension of size 1 at the specified position.
- **Permuting (`tensor.permute(dims_order)`):** Rearranges the dimensions of a tensor according to the specified order. This is useful, for example, when converting image formats (e.g., from Height-Width-Channel to Channel-Height-Width).

### 7. Tensor Indexing
- **PyTorch indexing is similar to NumPy indexing:**
    - Accessing specific elements using bracket notation (e.g., `tensor[0]`, `tensor[0][1][2]`).
    - Using `:` to select all values of a target dimension.
    - Multi-dimensional indexing (e.g., `tensor[:, :, 1]` to get all values of 0th and 1st dimensions but only index 1 of 2nd dimension).
    - Combining different indexing patterns to extract specific data from tensors.

### 8. PyTorch Tensors & NumPy Integration
- **Interoperability between PyTorch and NumPy:**
    - **NumPy array to PyTorch tensor:** `torch.from_numpy(array)`
    - **PyTorch tensor to NumPy array:** `tensor.numpy()`
- **Important note:** When converting from NumPy to PyTorch using `torch.from_numpy()`, the tensors share the same memory. However, changes to the original NumPy array after conversion may not always affect the tensor (depending on how the array is modified).

### 9. Reproducibility (Taking Random Out of Random)
- Understanding how neural networks learn: `start with random numbers -> tensor operations -> update random numbers to try and make them better representations of the data -> again -> again -> again...`
- **Random Seed Concept:** To reduce randomness in neural networks, PyTorch uses the concept of a **random seed** that "flavours" the randomness.
- **Setting Random Seeds:**
    - `torch.manual_seed(seed_value)` to ensure reproducible results
    - Creating multiple tensors with the same seed produces identical results
    - Essential for debugging and ensuring consistent experimental results

### 10. Running Tensors on GPUs
- **GPU Benefits:** GPUs provide faster computation on numbers thanks to CUDA + NVIDIA hardware + PyTorch working together.
- **Getting GPU Access:**
    1. **Easiest:** Use Google Colab for a free GPU (with upgrade options)
    2. **Own Hardware:** Use your own GPU
    3. **Cloud Computing:** Use services like GCP, AWS, Azure to rent cloud computers
- **GPU Setup:** For options 2 & 3, PyTorch + GPU drivers (CUDA) require setup. Refer to [PyTorch setup documentation](https://pytorch.org/get-started/locally/)
- **Checking GPU Availability:**
    - `torch.cuda.is_available()` to check if CUDA-compatible GPU is available
    - Setting up device-agnostic code for running on both CPU and GPU

The notebook provides practical code examples for each of these topics, allowing users to run the cells and see the outputs directly. It's a valuable resource for anyone starting their journey with PyTorch, covering all the essential tensor operations needed for deep learning.

## Running the Notebook

You can run this notebook in several ways:
1. **Google Colab** (Recommended): Click the "Open in Colab" badge above for free GPU access
2. **Local Jupyter**: Install PyTorch and run locally with `jupyter notebook`
3. **Other platforms**: Upload to any Jupyter-compatible environment

The notebook includes GPU availability checks and will work on both CPU and GPU environments.
