# AI/ML Project Collection

This repository contains implementations of various Artificial Intelligence and Machine Learning algorithms and techniques, including Convolutional Neural Networks (CNN), CUDA-accelerated Artificial Neural Networks (ANN), and Generative Adversarial Networks (GANs).

## Project Structure

### CNN/
Contains implementations of Convolutional Neural Networks for image classification tasks.
- `cat-dog-4 (6).ipynb`: Jupyter notebook implementing a CNN for cat vs dog image classification using PyTorch. Features include data loading, model training, and evaluation on the Dogs vs Cats dataset.

### CUDA-ANN/
CUDA implementations of Artificial Neural Networks for high-performance computing.
- `hello.cu`: CUDA C code implementing a neural network with batch normalization, ReLU activation, and softmax layers. Optimized for GPU acceleration.

### GANS/
Generative Adversarial Networks implementations.
- (Content to be added)

## Requirements

### CNN
- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Jupyter Notebook

### CUDA-ANN
- CUDA Toolkit
- cuBLAS
- cuRAND
- GCC/Clang compiler with CUDA support

### GANS
- (Requirements to be specified)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd final_taskphase
```

2. For CNN implementation:
```bash
pip install torch torchvision matplotlib jupyter
```

3. For CUDA-ANN:
Ensure CUDA toolkit is installed and properly configured on your system.

## Usage

### CNN Classification
1. Navigate to the CNN directory
2. Open the Jupyter notebook: `cat-dog-4 (6).ipynb`
3. Run the cells to train and evaluate the model
4. Note: The notebook expects data in `/kaggle/input/dogs-vs-cats/` format

### CUDA ANN
1. Navigate to CUDA-ANN directory
2. Compile the CUDA code:
```bash
nvcc hello.cu -o ann_cuda -lcublas -lcurand
```
3. Run the executable:
```bash
./ann_cuda
```

### GANs
(Usage instructions to be added)

## Features

- **CNN**: Image classification with custom CNN architecture
- **CUDA ANN**: GPU-accelerated neural network training
- **GANs**: Generative models for data synthesis

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Sarthak
