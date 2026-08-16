# zsl-mnist

A custom neural network built from scratch for MNIST digit classification, designed to showcase the capabilities of the [zsl](https://github.com/srmadrid/zsl) library in Zig.

This repository trains a multi-layer perceptron using stochastic gradient descent. It relies entirely on `zsl` for the underlying mathematical operations, matrix manipulations, and backward-pass gradient calculations via its tape-based automatic differentiation engine.

> ⚠️ **Note on Performance:** This project is a proof-of-concept rather than a production-grade machine learning framework. It prioritizes demonstrating the mechanics of `zsl` over raw training speed.

## Technical Details

Because the `zsl` autodiff engine is tape-based, memory consumption scales with the number of operations recorded. To maintain a minimal memory footprint, this implementation uses a batch size of 1 (updating weights after every single image) rather than mini-batching.

## Quick Start

Everything you need is included out of the box. The MNIST dataset images and labels are bundled in the local `data/` directory, and the Zig package manager will automatically fetch `zsl` during the build process.

### Prerequisites

- [Zig](https://ziglang.org/download/) (0.16.0)

### Building and Running

1. Clone this repository:

```bash
git clone [https://github.com/your-username/zsl-mnist.git](https://github.com/your-username/zsl-mnist.git)
cd zsl-mnist
```

2. Run the training loop. Compiling with the `ReleaseFast` optimization flag is highly recommended for reasonable execution times:

```bash
zig build run -Doptimize=ReleaseFast
```
