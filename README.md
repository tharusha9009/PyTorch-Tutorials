# PyTorch Tutorials
Comprehensive PyTorch tutorials for deep learning fundamentals

## PyTorch_Fundamentals.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tharusha9009/PyTorch-Tutorials/blob/main/Pytorch_Fundemenentals.ipynb)

This Jupyter notebook (`PyTorch_Fundamentals.ipynb`) serves as a comprehensive introduction to the fundamental concepts of PyTorch, with a primary focus on **tensors**. Tensors are multi-dimensional arrays that are the core data structure in PyTorch, similar to NumPy arrays but with the added benefit of GPU acceleration, which is crucial for deep learning applications.

**PyTorch Version Used:** 2.6.0+cu124  
**GPU Support:** Tesla T4 (15360MiB) with CUDA 12.4

The notebook provides hands-on, executable examples with real outputs covering the following essential PyTorch concepts:

---

## 📚 Table of Contents

### 1. Introduction to Tensors
Understanding the fundamental building blocks of PyTorch and their hierarchical structure:

- **Scalar (0-dimensional)**: Single number
  ```python
  scalar = torch.tensor(7)
  scalar.ndim  # 0
  scalar.item()  # 7 (extract Python number)
  ```

- **Vector (1-dimensional)**: Array of numbers
  ```python
  vector = torch.tensor([7, 7])
  vector.ndim  # 1
  vector.shape  # torch.Size([2])
  ```

- **Matrix (2-dimensional)**: 2D array of numbers
  ```python
  MATRIX = torch.tensor([[7, 8], [9, 8]])
  MATRIX.ndim  # 2
  MATRIX.shape  # torch.Size([2, 2])
  ```

- **Tensor (3+ dimensions)**: Higher-dimensional arrays
  ```python
  TENSOR = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])
  TENSOR.ndim  # 3
  TENSOR.shape  # torch.Size([1, 3, 3])
  ```

**Key Tensor Attributes:**
- `.ndim`: Number of dimensions
- `.shape`: Size of each dimension
- `.item()`: Extract Python number from scalar tensor

### 2. Random Tensors
**Why Random Tensors Matter:** Neural networks start with tensors full of random numbers and then adjust those random numbers to better represent the data through the learning process:

```
start with random numbers → look at data → update random numbers → look at data → update random numbers...
```

**Creating Random Tensors:**
- `torch.rand(3, 4)`: Create random tensor of shape (3, 4)
- `torch.rand(size=(3, 224, 224))`: Image-sized tensors (channels × height × width)
- `torch.rand(2, 4, 5)`: Create 3D random tensor
- `torch.rand(size=(224, 224, 3))`: Alternative image format (height × width × channels)

### 3. Creating Tensors with Zeros and Ones
Specialized tensor creation methods for initialization:

- **Zeros:** `torch.zeros(size=(3, 4))` - Tensors filled with zeros
- **Ones:** `torch.ones(size=(3, 4))` - Tensors filled with ones  
- **Range:** `torch.arange(0, 10, 2)` - Creates: `tensor([0, 2, 4, 6, 8])`
- **Like Operations:**
  - `torch.zeros_like(input=torch.arange(0, 10))`: Zeros with same shape as input

### 4. Tensor Datatypes and Attributes
**The 3 Most Common PyTorch Errors:**
1. **Tensors not right datatype** → Check with `tensor.dtype`
2. **Tensors not right shape** → Check with `tensor.shape`
3. **Tensors not on the right device** → Check with `tensor.device`

**Working with Datatypes:**
- Specify during creation: `torch.tensor([3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False)`
- Convert existing tensors: `tensor.type(torch.float16)`
- Mix different types: `float_32_tensor * int_32_tensor`

**Getting Tensor Information:**
```python
some_tensor = torch.rand(3, 4)
print(f"Datatype of tensor: {some_tensor.dtype}")      # torch.float32
print(f"Shape of tensor: {some_tensor.shape}")          # torch.Size([3, 4])
print(f"Device tensor is on: {some_tensor.device}")     # cpu
```

### 5. Tensor Operations and Manipulations
**Basic Arithmetic Operations:**
```python
tensor = torch.tensor([1, 2, 3])
tensor + 10      # Element-wise addition
tensor * 10      # Element-wise multiplication  
tensor - 10      # Element-wise subtraction

# PyTorch functions
torch.add(tensor, 10)
torch.mul(tensor, 10)
```

**Matrix Multiplication Rules:**
Two main types of multiplication in neural networks:

1. **Element-wise multiplication:** `tensor * tensor` (Hadamard product)
   ```python
   # Example output: tensor([0, 10, 20]) * tensor([0, 10, 20]) = tensor([0, 100, 400])
   ```

2. **Matrix multiplication:** `torch.matmul(tensor, tensor)`, `tensor @ tensor`, or `torch.mm(tensor, tensor)`

**Matrix Multiplication Requirements:**
1. **Inner dimensions must match:**
   - ✅ `(2, 3) @ (3, 2)` → Valid
   - ❌ `(3, 2) @ (3, 2)` → Invalid
2. **Result shape = outer dimensions:**
   - `(2, 3) @ (3, 2) → (2, 2)`
   - `(3, 2) @ (2, 3) → (3, 3)`

**Solving Shape Errors:**
```python
tensor_A = torch.tensor([[1, 2], [3, 4], [5, 6]])  # Shape: (3, 2)
tensor_B = torch.tensor([[7, 10], [8, 11], [9, 12]])  # Shape: (3, 2)

# Use transpose to fix dimension mismatch
output = torch.matmul(tensor_A, tensor_B.T)  # (3, 2) @ (2, 3) → (3, 3)
```

### 6. Tensor Aggregations (Min, Max, Mean, Sum)
**Statistical Operations:**
```python
x = torch.arange(0, 100, 10)  # tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# Basic aggregations
torch.min(x), x.min()           # Minimum values
torch.max(x), x.max()           # Maximum values  
torch.mean(x.type(torch.float32))  # Mean (requires float32)
torch.sum(x), x.sum()           # Sum of all elements
```

**Positional Operations:**
```python
x1 = torch.arange(1, 100, 10)  # tensor([1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
x1.argmin()  # Index of minimum value
x1.argmax()  # Index of maximum value  
```

### 7. Reshaping, Stacking, Squeezing, and Unsqueezing
**Essential Shape Manipulation Operations:**

```python
x = torch.arange(1., 10.)  # tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])

# Reshape: Changes shape (total elements must remain same)
x_reshaped = x.reshape(1, 9)  # torch.Size([1, 9])

# View: Returns new tensor sharing memory with original
z = x.view(1, 9)  # Changes to z affect x!

# Stack: Combine tensors along new dimension  
x_stacked = torch.stack([x, x, x, x], dim=0)  # torch.Size([4, 9])

# Squeeze: Remove dimensions of size 1
x_squeezed = x_reshaped.squeeze()  # torch.Size([9])

# Unsqueeze: Add dimension of size 1
x_unsqueezed = x_squeezed.unsqueeze(dim=0)  # torch.Size([1, 9])

# Permute: Rearrange dimensions (useful for image processing)
x_original = torch.rand(size=(224, 224, 3))  # [height, width, color_channels]
x_permuted = x_original.permute(2, 0, 1)     # [color_channels, height, width]
```

### 8. Tensor Indexing
**PyTorch indexing follows NumPy conventions:**
```python
x = torch.arange(1, 10).reshape(1, 3, 3)

# Basic indexing
x[0]          # First dimension
x[0][0]       # First two dimensions  
x[0][1][2]    # Specific element

# Advanced indexing with ":"
x[:, 0]       # All of first dim, index 0 of second dim
x[:, :, 1]    # All of first two dims, index 1 of third dim
x[:, 1, 1]    # All of first dim, index 1 of second and third dims
x[0, 0, :]    # Index 0 of first two dims, all of third dim

# Example outputs
x[0][2][2]    # Returns: tensor(9)
x[:, :, 2]    # Returns: tensor([[3, 6, 9]])
```

### 9. PyTorch & NumPy Integration
**Seamless Interoperability:**
```python
# NumPy → PyTorch
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)

# PyTorch → NumPy  
tensor = torch.ones(8)
numpy_tensor = tensor.numpy()
```

**⚠️ Important Memory Considerations:**
- `torch.from_numpy()` may create tensors that share memory with the original NumPy array
- Changes to one may affect the other depending on the operation
- Always check behavior when modifying converted arrays/tensors

### 10. Reproducibility (Taking Random Out of Random)
**Neural Network Learning Process:**
```
start with random numbers → tensor operations → update random numbers → repeat...
```

**Random Seeds for Reproducible Results:**
```python
# Without seed - different results each time
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A == random_tensor_B)  # All False

# With seed - reproducible results
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)  
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C == random_tensor_D)  # All True
```

### 11. Running Tensors on GPUs
**GPU Acceleration Benefits:**
- Dramatically faster computation through CUDA + NVIDIA hardware + PyTorch integration
- Essential for training large neural networks

**Current Setup:** Tesla T4 GPU with 15360MiB memory and CUDA 12.4

**GPU Availability Check:**
```python
# Check for GPU access
torch.cuda.is_available()      # True (if GPU available)
torch.cuda.device_count()      # Number of available GPUs
```

**Device-Agnostic Code Setup:**
```python
# Best practice: Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)  # "cuda" (if GPU available)
```

**Moving Tensors Between Devices:**
```python
# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])
print(tensor.device)  # cpu

# Move to GPU
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu.device)  # cuda:0

# Important: GPU tensors can't directly convert to NumPy
# tensor_on_gpu.numpy()  # This will error!

# Solution: Move back to CPU first
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
```

**⚠️ GPU-NumPy Compatibility:**
- Tensors on GPU cannot be directly converted to NumPy arrays
- Must move tensor to CPU first: `tensor_on_gpu.cpu().numpy()`

---

## 🚀 Getting Started

1. **Open in Google Colab** - Click the Colab badge above for immediate GPU access
2. **Local Setup** - Clone this repository and ensure you have PyTorch 2.6.0+ with CUDA installed
3. **Run the Notebook** - Execute cells sequentially to follow along with the tutorial

## 📋 Prerequisites

- Basic Python knowledge
- Understanding of NumPy arrays (helpful but not required)  
- Mathematical concepts: vectors, matrices, basic linear algebra
- Google Colab account (for GPU access) or local CUDA setup

## 🎯 Learning Outcomes

After completing this notebook, you will:
- ✅ Understand PyTorch tensor fundamentals and operations
- ✅ Know how to create, manipulate, and operate on tensors
- ✅ Be familiar with the 3 most common PyTorch errors and how to debug them
- ✅ Have hands-on experience with tensor operations used in deep learning
- ✅ Understand the relationship between PyTorch and NumPy
- ✅ Know how to set up device-agnostic code for CPU/GPU compatibility
- ✅ Be able to move tensors between CPU and GPU
- ✅ Understand reproducibility in neural networks
- ✅ Be prepared to move on to more advanced PyTorch topics like neural networks

## 🔧 Technical Specifications

- **PyTorch Version:** 2.6.0+cu124
- **GPU:** Tesla T4 (15360MiB VRAM)
- **CUDA Version:** 12.4
- **Driver Version:** 550.54.15
- **Additional Libraries:** pandas, numpy, matplotlib

---

**Note:** This notebook contains executable code examples with actual outputs and performance measurements (using `%%time` magic commands), making it perfect for hands-on learning. Each concept is demonstrated with practical examples that you can run and modify to deepen your understanding. The notebook includes working GPU implementation with real device switching examples.
