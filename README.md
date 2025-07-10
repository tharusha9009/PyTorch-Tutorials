# PyTorch-Tutorials
PyTorch tutorials 

## Pytorch_Fundemenentals.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tharusha9009/PyTorch-Tutorials/blob/main/Pytorch_Fundemenentals.ipynb)

This Jupyter notebook (`Pytorch_Fundemenentals.ipynb`) serves as a comprehensive introduction to the fundamental concepts of PyTorch, with a primary focus on **tensors**. Tensors are multi-dimensional arrays that are the core data structure in PyTorch, similar to NumPy arrays but with the added benefit of GPU acceleration, which is crucial for deep learning.

**PyTorch Version Used:** 2.6.0+cu124

The notebook walks through the following key areas:

### 1. Introduction to Tensors
- **What are Tensors?** Understanding the building blocks of PyTorch:
  - **Scalar (0D)**: Single number - `torch.tensor(7)`
  - **Vector (1D)**: Array of numbers - `torch.tensor([7,7])`  
  - **Matrix (2D)**: 2D array of numbers - `torch.tensor([[7,8],[9,8]])`
  - **Tensor (3D+)**: Higher-dimensional arrays - `torch.tensor([[[1,2,3],[3,6,9],[2,4,5]]])`
- **Tensor Attributes:**
  - `.ndim`: Number of dimensions
  - `.shape`: Size of each dimension  
  - `.item()`: Get the Python number from a scalar tensor

### 2. Random Tensors
- **Why Random Tensors?** Neural networks start with tensors full of random numbers and then adjust those random numbers to better represent the data through the learning process:
  - `start with random numbers → look at data → update random numbers → look at data → update random numbers`
- **Creating Random Tensors:**
  - `torch.rand(size)`: Random numbers uniformly distributed between 0 and 1
  - `torch.rand(3,4)`: Create random tensor of specific shape
  - `torch.rand(size=(3,224,224))`: Create image-sized tensors (common in computer vision)

### 3. Creating Tensors with Zeros and Ones
- **Specialized Tensor Creation:**
  - `torch.zeros(size=(3,4))`: Tensors filled with zeros
  - `torch.ones(size=(3,4))`: Tensors filled with ones
  - `torch.arange(start, end, step)`: Tensors with a sequence of numbers
  - `torch.zeros_like(input_tensor)`, `torch.ones_like(input_tensor)`: Tensors with the same shape as an existing tensor

### 4. Tensor Datatypes
- **The 3 Big Errors in PyTorch & Deep Learning:**
  1. **Tensors not right datatype** - Check with `tensor.dtype`
  2. **Tensors not right shape** - Check with `tensor.shape`  
  3. **Tensors not on the right device** - Check with `tensor.device`
- **Working with Datatypes:**
  - Specifying `dtype` during creation: `torch.tensor([3.0,6.0,9.0], dtype=torch.float32)`
  - Converting datatypes: `tensor.type(torch.float16)`
  - Common datatypes: `torch.float32`, `torch.float16`, `torch.long`
- **Device Information:**
  - Check device location: `tensor.device` (e.g., 'cpu', 'cuda')

### 5. Tensor Manipulations and Operations
- **Basic Arithmetic:**
  - Element-wise operations: `tensor + 10`, `tensor * 10`, `tensor - 10`
  - PyTorch functions: `torch.add(tensor, 10)`, `torch.mul(tensor, 10)`
- **Matrix Multiplication:**
  - **Element-wise multiplication:** `tensor * tensor` 
  - **Matrix multiplication:** `torch.matmul(tensor, tensor)`, `tensor @ tensor`, or `torch.mm(tensor, tensor)`
  - **Matrix Multiplication Rules:**
    1. **Inner dimensions must match:** `(2,3) @ (3,2)` ✓, `(3,2) @ (3,2)` ✗
    2. **Resulting shape = outer dimensions:** `(2,3) @ (3,2) → (2,2)`
  - **Solving Shape Errors:**
    - Use `.T` or `tensor.transpose()` to transpose matrices
    - Example: `torch.matmul(tensor_A, tensor_B.T)` when dimensions don't align

### 6. Tensor Aggregations (Finding Min, Max, Mean, Sum)
- **Statistical Operations:**
  - Minimum: `torch.min(tensor)` or `tensor.min()`
  - Maximum: `torch.max(tensor)` or `tensor.max()`
  - Mean: `torch.mean(tensor.type(torch.float32))` or `tensor.type(torch.float32).mean()`
    - **Note:** `mean()` requires floating-point input
  - Sum: `torch.sum(tensor)` or `tensor.sum()`
- **Positional Min/Max:**
  - Index of minimum: `tensor.argmin()`
  - Index of maximum: `tensor.argmax()`

### 7. Reshaping, Stacking, Squeezing, and Unsqueezing
- **Reshaping Operations:**
  - **Reshape:** `tensor.reshape(new_shape)` - Changes shape (total elements must remain same)
  - **View:** `tensor.view(new_shape)` - Returns new tensor sharing same memory as original
  - **Stack:** `torch.stack([tensor1, tensor2, tensor3], dim=0)` - Combine tensors along new dimension
  - **Squeeze:** `tensor.squeeze()` - Removes all dimensions of size 1
  - **Unsqueeze:** `tensor.unsqueeze(dim=0)` - Adds dimension of size 1 at specified position
  - **Permute:** `tensor.permute(2,0,1)` - Rearranges dimensions (useful for image format conversion)
    - Example: Convert `(height, width, channels)` to `(channels, height, width)`

### 8. Tensor Indexing
- **PyTorch indexing follows NumPy conventions:**
  - Access elements: `tensor[0]`, `tensor[0][1][2]`
  - Select all of a dimension: `tensor[:,0]` 
  - Multi-dimensional indexing: `tensor[:,:,1]` (all of 0th and 1st dims, index 1 of 2nd dim)
  - Complex indexing: `tensor[:,1,1]`, `tensor[0,0,:]`

### 9. PyTorch Tensors & NumPy Integration
- **Interoperability:**
  - **NumPy → PyTorch:** `torch.from_numpy(array)`
  - **PyTorch → NumPy:** `tensor.numpy()`
- **Important Memory Considerations:**
  - `torch.from_numpy()` creates tensors that may share memory with original NumPy array
  - Changes to converted tensors/arrays may affect each other depending on the operation

### 10. Reproducibility (Taking Random Out of Random)
- **Neural Network Learning Process:**
  - `start with random numbers → tensor operations → update random numbers → repeat...`
- **Random Seeds for Reproducible Results:**
  - Set seed: `torch.manual_seed(42)`
  - Creates "flavored" randomness - same seed produces same "random" numbers
  - Essential for reproducible experiments and debugging
- **Example:**
  ```python
  RANDOM_SEED = 42
  torch.manual_seed(RANDOM_SEED)
  tensor_A = torch.rand(3,4)
  torch.manual_seed(RANDOM_SEED)  
  tensor_B = torch.rand(3,4)
  # tensor_A == tensor_B will be True
  ```

### 11. Running Tensors on GPUs
- **GPU Acceleration Benefits:**
  - Faster computation thanks to CUDA + NVIDIA hardware + PyTorch integration
- **GPU Access Options:**
  1. **Google Colab** - Free GPU access (easiest option)
  2. **Personal GPU** - Requires CUDA setup
  3. **Cloud Computing** - AWS, GCP, Azure (requires setup)
- **Checking GPU Availability:**
  - `torch.cuda.is_available()` - Returns True if GPU is accessible
  - `nvidia-smi` - Command to check GPU status (if available)
- **Device-Agnostic Code:** Setting up code that works on both CPU and GPU

---

This notebook provides hands-on code examples for each topic, allowing users to run the cells and see outputs directly. It serves as a comprehensive foundation for anyone beginning their PyTorch journey, covering all essential tensor operations needed for deep learning applications.
