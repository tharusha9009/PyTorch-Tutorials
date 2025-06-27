# PyTorch-Tutorials
PyTorch tutorials 

## Pytorch_Fundemenentals.ipynb

This Jupyter notebook (`Pytorch_Fundemenentals.ipynb`) serves as an introduction to the fundamental concepts of PyTorch, with a primary focus on **tensors**. Tensors are multi-dimensional arrays that are the core data structure in PyTorch, similar to NumPy arrays but with the added benefit of GPU acceleration, which is crucial for deep learning.

The notebook walks through the following key areas:

### 1. Tensor Creation and Basics
- **What are Tensors?** Introduction to scalars (0D), vectors (1D), matrices (2D), and higher-dimensional tensors.
- **Creating Tensors:**
    - Using `torch.tensor()` to create tensors from Python lists or numerical values.
    - Understanding tensor attributes:
        - `.ndim`: Number of dimensions.
        - `.shape`: Size of each dimension.
        - `.item()`: Get the Python number from a scalar tensor.
- **Specialized Tensor Creation:**
    - `torch.rand(size)`: Tensors with random numbers uniformly distributed between 0 and 1.
    - `torch.zeros(size)`: Tensors filled with zeros.
    - `torch.ones(size)`: Tensors filled with ones.
    - `torch.arange(start, end, step)`: Tensors with a sequence of numbers.
    - `torch.zeros_like(input_tensor)`, `torch.ones_like(input_tensor)`: Tensors with the same shape as an existing tensor, filled with zeros or ones.

### 2. Tensor Datatypes
- Importance of using the correct datatype for tensors (e.g., `torch.float32`, `torch.float16`, `torch.int64`).
- Common issues arising from mismatched datatypes in operations.
- Specifying `dtype` during tensor creation (e.g., `torch.tensor([1, 2, 3], dtype=torch.float32)`).
- Changing tensor datatype using the `.type(new_dtype)` method.
- Checking tensor datatype with `tensor.dtype`.
- Checking the device a tensor is on with `tensor.device` (e.g., 'cpu', 'cuda').

### 3. Tensor Manipulations and Operations
- **Basic Arithmetic:**
    - Element-wise addition, subtraction, multiplication, and division.
    - Using both overloaded operators (e.g., `tensor + 10`) and PyTorch functions (e.g., `torch.add(tensor, 10)`).
- **Matrix Multiplication:**
    - Distinction between element-wise multiplication (`*`) and matrix multiplication (`torch.matmul()` or `@` or `torch.mm()`).
    - Rules for matrix multiplication:
        - Inner dimensions must match (e.g., `(a, b) @ (b, c)`).
        - The resulting matrix shape is determined by the outer dimensions (e.g., `(a, c)`).
    - Using `.T` or `tensor.transpose()` to transpose matrices to satisfy shape requirements for multiplication.
- **Tensor Aggregations:**
    - Finding minimum (`torch.min(tensor)` or `tensor.min()`).
    - Finding maximum (`torch.max(tensor)` or `tensor.max()`).
    - Calculating the mean (`torch.mean(tensor.type(torch.float32))` or `tensor.type(torch.float32).mean()`). Note: `mean()` typically requires floating-point input.
    - Summing all elements (`torch.sum(tensor)` or `tensor.sum()`).
- **Positional Min/Max:**
    - Finding the index of the minimum value (`tensor.argmin()`).
    - Finding the index of the maximum value (`tensor.argmax()`).

### 4. Reshaping, Viewing, Stacking, Squeezing, and Unsqueezing
- **Reshaping (`tensor.reshape(new_shape)`):** Changes the shape of a tensor. The total number of elements must remain the same.
- **Viewing (`tensor.view(new_shape)`):** Returns a new tensor with the desired shape that shares the same underlying data (memory) as the original tensor. Modifying the view will modify the original tensor.
- **Stacking (`torch.stack(list_of_tensors, dim)`):** Concatenates a sequence of tensors along a new dimension.
- **Squeezing (`tensor.squeeze()`):** Removes all dimensions of size 1 from a tensor.
- **Unsqueezing (`tensor.unsqueeze(dim)`):** Adds a dimension of size 1 at the specified position.
- **Permuting (`tensor.permute(dims_order)`):** Rearranges the dimensions of a tensor according to the specified order. This is useful, for example, when converting image formats (e.g., from Height-Width-Channel to Channel-Height-Width).

The notebook provides practical code examples for each of these topics, allowing users to run the cells and see the outputs directly. It's a valuable resource for anyone starting their journey with PyTorch.
