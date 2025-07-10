# PyTorch Tutorials
Comprehensive PyTorch tutorials for deep learning fundamentals

## PyTorch_Fundamentals.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tharusha9009/PyTorch-Tutorials/blob/main/Pytorch_Fundemenentals.ipynb)

This Jupyter notebook (`PyTorch_Fundamentals.ipynb`) serves as a comprehensive introduction to the fundamental concepts of PyTorch, with a primary focus on **tensors**. Tensors are multi-dimensional arrays that are the core data structure in PyTorch, similar to NumPy arrays but with the added benefit of GPU acceleration, which is crucial for deep learning applications.

**PyTorch Version Used:** 2.6.0+cu124

The notebook provides hands-on, executable examples covering the following essential PyTorch concepts:

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
  matrix = torch.tensor([[7, 8], [9, 8]])
  matrix.ndim  # 2
  matrix.shape  # torch.Size([2, 2])
  ```

- **Tensor (3+ dimensions)**: Higher-dimensional arrays
  ```python
  tensor = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])
  tensor.ndim  # 3
  tensor.shape  # torch.Size([1, 3, 3])
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
- `torch.rand(size)`: Random numbers uniformly distributed between 0 and 1
- `torch.rand(3, 4)`: Create random tensor of shape (3, 4)
- `torch.rand(size=(3, 224, 224))`: Create image-sized tensors (height × width × channels)
- `torch.rand(2, 4, 5)`: Create 3D random tensor

### 3. Creating Tensors with Zeros and Ones
Specialized tensor creation methods for initialization:

- **Zeros:** `torch.zeros(size=(3, 4))` - Tensors filled with zeros
- **Ones:** `torch.ones(size=(3, 4))` - Tensors filled with ones
- **Range:** `torch.arange(start, end, step)` - Tensors with sequences
  ```python
  torch.arange(0, 10, 2)  # tensor([0, 2, 4, 6, 8])
  ```
- **Like Operations:**
  - `torch.zeros_like(input_tensor)`: Zeros with same shape as input
  - `torch.ones_like(input_tensor)`: Ones with same shape as input

### 4. Tensor Datatypes and Attributes
**The 3 Most Common PyTorch Errors:**
1. **Tensors not right datatype** → Check with `tensor.dtype`
2. **Tensors not right shape** → Check with `tensor.shape`
3. **Tensors not on the right device** → Check with `tensor.device`

**Working with Datatypes:**
- Specify during creation: `torch.tensor([3.0, 6.0, 9.0], dtype=torch.float32)`
- Convert existing tensors: `tensor.type(torch.float16)`
- Common datatypes: `torch.float32`, `torch.float16`, `torch.long`

**Getting Tensor Information:**
```python
some_tensor = torch.rand(3, 4)
print(f"Datatype: {some_tensor.dtype}")      # torch.float32
print(f"Shape: {some_tensor.shape}")          # torch.Size([3, 4])
print(f"Device: {some_tensor.device}")        # cpu
```

### 5. Tensor Operations and Manipulations
**Basic Arithmetic Operations:**
- Element-wise operations: `tensor + 10`, `tensor * 10`, `tensor - 10`
- PyTorch functions: `torch.add(tensor, 10)`, `torch.mul(tensor, 10)`

**Matrix Multiplication Rules:**
Two main types of multiplication in neural networks:

1. **Element-wise multiplication:** `tensor * tensor` (Hadamard product)
2. **Matrix multiplication:** `torch.matmul(tensor, tensor)`, `tensor @ tensor`, or `torch.mm(tensor, tensor)`

**Matrix Multiplication Requirements:**
1. **Inner dimensions must match:**
   - ✅ `(2, 3) @ (3, 2)` → Valid
   - ❌ `(3, 2) @ (3, 2)` → Invalid
2. **Result shape = outer dimensions:**
   - `(2, 3) @ (3, 2) → (2, 2)`
   - `(3, 2) @ (2, 3) → (3, 3)`

**Solving Shape Errors:**
- Use `.T` or `tensor.transpose()` to transpose matrices
- Example: `torch.matmul(tensor_A, tensor_B.T)` when dimensions don't align

### 6. Tensor Aggregations (Min, Max, Mean, Sum)
**Statistical Operations:**
- **Minimum:** `torch.min(tensor)` or `tensor.min()`
- **Maximum:** `torch.max(tensor)` or `tensor.max()`
- **Mean:** `torch.mean(tensor.type(torch.float32))` or `tensor.type(torch.float32).mean()`
  - ⚠️ **Note:** `mean()` requires floating-point input
- **Sum:** `torch.sum(tensor)` or `tensor.sum()`

**Positional Operations:**
- **Index of minimum:** `tensor.argmin()`
- **Index of maximum:** `tensor.argmax()`

### 7. Reshaping, Stacking, Squeezing, and Unsqueezing
**Essential Shape Manipulation Operations:**

- **Reshape:** `tensor.reshape(new_shape)` - Changes shape (total elements must remain same)
- **View:** `tensor.view(new_shape)` - Returns new tensor sharing memory with original
- **Stack:** `torch.stack([tensor1, tensor2, tensor3], dim=0)` - Combine tensors along new dimension
- **Squeeze:** `tensor.squeeze()` - Removes all dimensions of size 1
- **Unsqueeze:** `tensor.unsqueeze(dim=0)` - Adds dimension of size 1 at specified position
- **Permute:** `tensor.permute(2, 0, 1)` - Rearranges dimensions
  - Example: Convert `(height, width, channels)` to `(channels, height, width)`

### 8. Tensor Indexing
**PyTorch indexing follows NumPy conventions:**
- Access elements: `tensor[0]`, `tensor[0][1][2]`
- Select all of a dimension: `tensor[:, 0]`
- Multi-dimensional indexing: `tensor[:, :, 1]` (all of 0th and 1st dims, index 1 of 2nd dim)
- Complex indexing: `tensor[:, 1, 1]`, `tensor[0, 0, :]`

### 9. PyTorch & NumPy Integration
**Seamless Interoperability:**
- **NumPy → PyTorch:** `torch.from_numpy(numpy_array)`
- **PyTorch → NumPy:** `tensor.numpy()`

**⚠️ Important Memory Considerations:**
- `torch.from_numpy()` may create tensors that share memory with the original NumPy array
- Changes to converted tensors/arrays may affect each other depending on the operation

### 10. Reproducibility (Taking Random Out of Random)
**Neural Network Learning Process:**
```
start with random numbers → tensor operations → update random numbers → repeat...
```

**Random Seeds for Reproducible Results:**
- Set seed: `torch.manual_seed(42)`
- Creates "flavored" randomness - same seed produces same "random" numbers
- Essential for reproducible experiments and debugging

**Example:**
```python
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
tensor_A = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
tensor_B = torch.rand(3, 4)

# tensor_A == tensor_B will be True element-wise
```

### 11. Running Tensors on GPUs (Coming Soon)
**GPU Acceleration Benefits:**
- Dramatically faster computation through CUDA + NVIDIA hardware + PyTorch integration
- Essential for training large neural networks

**GPU Access Options:**
1. **Google Colab** - Free GPU access (easiest for beginners)
2. **Personal GPU** - Requires CUDA setup
3. **Cloud Computing** - AWS, GCP, Azure (requires setup and billing)

**Checking GPU Availability:**
- `torch.cuda.is_available()` - Returns True if GPU is accessible
- `nvidia-smi` - Command-line tool to check GPU status
- **Device-agnostic code:** Writing code that works on both CPU and GPU

---

## 🚀 Getting Started

1. **Open in Google Colab** - Click the Colab badge above for immediate access
2. **Local Setup** - Clone this repository and ensure you have PyTorch 2.6.0+ installed
3. **Run the Notebook** - Execute cells sequentially to follow along with the tutorial

## 📋 Prerequisites

- Basic Python knowledge
- Understanding of NumPy arrays (helpful but not required)
- Mathematical concepts: vectors, matrices, basic linear algebra

## 🎯 Learning Outcomes

After completing this notebook, you will:
- Understand PyTorch tensor fundamentals and operations
- Know how to create, manipulate, and operate on tensors
- Be familiar with common PyTorch errors and how to debug them
- Have hands-on experience with tensor operations used in deep learning
- Understand the relationship between PyTorch and NumPy
- Be prepared to move on to more advanced PyTorch topics like neural networks

---

**Note:** This notebook contains executable code examples with outputs, making it perfect for hands-on learning. Each concept is demonstrated with practical examples that you can run and modify to deepen your understanding.
