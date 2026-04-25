# CUDA-ANN: High-Performance Neural Network Implementation

This directory contains a CUDA-accelerated Artificial Neural Network (ANN) implementation optimized for GPU computing. The implementation demonstrates advanced deep learning techniques with high-performance computing capabilities.

## Overview

The `hello.cu` file implements a complete neural network training pipeline using NVIDIA CUDA, cuBLAS, and cuRAND libraries. The network is designed for handwritten digit recognition using the MNIST dataset, achieving efficient training through GPU acceleration.

## Architecture

### Network Structure
- **Input Layer**: 784 neurons (28×28 MNIST images)
- **Hidden Layer 1**: 64 neurons with Batch Normalization + Leaky ReLU + Dropout
- **Hidden Layer 2**: 32 neurons with Batch Normalization + Leaky ReLU + Dropout
- **Output Layer**: 10 neurons with Softmax activation

### Key Features

#### GPU Acceleration
- **cuBLAS Integration**: Leverages NVIDIA's Basic Linear Algebra Subprograms for optimized matrix operations
- **CUDA Kernels**: Custom GPU kernels for activation functions, batch normalization, and loss computation
- **Memory Management**: Efficient GPU memory allocation and data transfer

#### Advanced Techniques
- **Batch Normalization**: Normalizes layer inputs for stable training and faster convergence
- **Leaky ReLU Activation**: Prevents dead neurons with small negative slope (α=0.01)
- **Dropout Regularization**: 30% dropout rate during training to prevent overfitting
- **Softmax Output**: Converts logits to probability distributions
- **Cross-Entropy Loss**: Standard loss function for multi-class classification

#### Optimization
- **Momentum SGD**: Stochastic Gradient Descent with momentum (μ=0.9)
- **Learning Rate**: 0.0005 with constant schedule
- **Batch Size**: 16 samples per training iteration
- **Data Shuffling**: Random shuffling between epochs

## Dependencies

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 3.0+)
- Sufficient GPU memory for model parameters and batch processing

### Software Requirements
- **CUDA Toolkit**: Version 10.0 or later
- **cuBLAS**: Included with CUDA Toolkit
- **cuRAND**: Included with CUDA Toolkit
- **GCC/Clang**: C/C++ compiler with CUDA support

### Dataset
- **MNIST Dataset**: Pre-processed binary files
  - Training: `x_train2.bin` (features), `y_train2.bin` (labels)
  - Testing: `x_test2.bin` (features), `y_test2.bin` (labels)
- **Data Format**: Binary files with float32 features and uint8 labels

## Installation & Setup

1. **Install CUDA Toolkit**:
   ```bash
   # Download and install from NVIDIA website
   # Or use package manager (Ubuntu/Debian):
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   ```

2. **Verify Installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Prepare Dataset**:
   Place the MNIST binary files in the `Downloads/` directory relative to the executable:
   ```
   Downloads/
   ├── x_train2.bin
   ├── y_train2.bin
   ├── x_test2.bin
   └── y_test2.bin
   ```

## Compilation

Compile the CUDA code with optimization flags:

```bash
nvcc hello.cu -o cuda_ann \
     -lcublas -lcurand \
     -O3 \
     -arch=sm_60 \
     -std=c++11
```

### Compiler Flags Explanation
- `-lcublas`: Link cuBLAS library
- `-lcurand`: Link cuRAND library
- `-O3`: Maximum optimization level
- `-arch=sm_60`: Target Pascal architecture (adjust for your GPU)
- `-std=c++11`: Use C++11 standard

## Usage

### Running the Training
```bash
./cuda_ann
```

### Expected Output
```
Epoch 1: Train acc = XX.XX%
Epoch 2: Train acc = XX.XX%
...
Epoch 10: Train acc = XX.XX%
Test acc = XX.XX%
```

### Performance Notes
- **Training Time**: Approximately 2-5 minutes per epoch on modern GPUs
- **Memory Usage**: ~50-100MB GPU memory
- **Accuracy**: Expected test accuracy >95% after 10 epochs

## Code Structure

### Core Components

#### Data Structures
- `Dataset`: Holds training/testing data and metadata
- `NeuralNetwork`: Main network structure with layers and parameters
- `BatchNormParams`: Batch normalization parameters per layer
- `LayerParams`: Weights and biases for each layer

#### CUDA Kernels
- `Relu`: Leaky ReLU activation function
- `softmaxForward`: Softmax computation for output layer
- `batchNorm`: Batch normalization (forward and backward)
- `dropoutForward`: Dropout regularization
- `crossEntropyLoss`: Loss computation and gradient calculation

#### Host Functions
- `loadDataset`: Loads binary MNIST data files
- `normalizeData`: Feature normalization (zero mean, unit variance)
- `initNeuralNetwork`: Network initialization with Xavier weight initialization
- `forwardPropagate`: Forward pass through the network
- `backwardPropagate`: Backward pass with gradient computation
- `calculateAccuracy`: Evaluation on test/validation data

### Memory Management
- **GPU Memory**: All computations performed on GPU
- **Data Transfer**: Minimal host-device transfers for efficiency
- **Cleanup**: Proper memory deallocation to prevent leaks

## Training Details

### Hyperparameters
```c
int n_epochs = 10;
int batch_size = 16;
float lr = 0.0005;
float momentum = 0.9;
float dropout_rate = 0.3;
```

### Data Preprocessing
- **Normalization**: Z-score normalization per feature
- **Shuffling**: Random sample order each epoch
- **Batching**: Mini-batch gradient descent

### Training Loop
1. Load and preprocess data
2. Initialize network parameters
3. For each epoch:
   - Shuffle training data
   - Process mini-batches
   - Forward pass → Loss computation → Backward pass → Parameter update
   - Evaluate training accuracy
4. Final test evaluation

## Performance Optimization

### cuBLAS Usage
- Matrix multiplication: `cublasSgemm_v2`
- Vector operations: `cublasSaxpy_v2`, `cublasSscal_v2`
- Memory-efficient BLAS operations

### CUDA Best Practices
- **Kernel Launch Configuration**: Optimized block/thread dimensions
- **Memory Coalescing**: Efficient global memory access patterns
- **Shared Memory**: Not used (could be optimization opportunity)
- **Atomic Operations**: Used for loss accumulation

### Potential Improvements
- **Mixed Precision**: FP16 training for speed
- **Multi-GPU**: Data parallelism across multiple GPUs
- **Tensor Cores**: Utilize Tensor Core operations
- **Memory Pooling**: Reuse allocated memory

## Troubleshooting

### Common Issues

#### Compilation Errors
- Ensure CUDA toolkit is properly installed
- Check GPU compute capability compatibility
- Verify compiler paths and library links

#### Runtime Errors
- **CUDA Error**: Check GPU memory availability
- **File Not Found**: Verify dataset file paths
- **Memory Allocation Failed**: Reduce batch size or model size

#### Performance Issues
- **Slow Training**: Check GPU utilization with `nvidia-smi`
- **Low Accuracy**: Verify data preprocessing and hyperparameters
- **Memory Errors**: Monitor GPU memory usage

### Debug Mode
Add debug flags for troubleshooting:
```bash
nvcc hello.cu -o cuda_ann -lcublas -lcurand -g -G
```

## References

- **CUDA Documentation**: https://docs.nvidia.com/cuda/
- **cuBLAS Guide**: https://docs.nvidia.com/cuda/cublas/
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **Batch Normalization**: https://arxiv.org/abs/1502.03167

## License

This implementation is provided for educational and research purposes. Please cite appropriately if used in academic work.

## Author

Sarthak - CUDA Neural Network Implementation
