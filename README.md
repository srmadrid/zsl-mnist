# zsl-mnist

A custom neural network built from scratch for MNIST digit classification, designed to showcase the capabilities of the [zsl](https://github.com/srmadrid/zsl) library in Zig.

This repository trains a multi-layer perceptron using stochastic gradient descent. It relies entirely on `zsl` for the underlying mathematical operations, matrix manipulations, and backward-pass gradient calculations via its tape-based automatic differentiation engine.

> ⚠️ **Note on Performance:** This project is a proof-of-concept rather than a production-grade machine learning framework. It prioritizes demonstrating the mechanics of `zsl` over training speed.

## Technical Details

This project implements a standard 2-layer Multi-Layer Perceptron (MLP) featuring a hidden layer with ReLU as the activation function, and an output layer mapping to the 10 possible MNIST digit classes.

### Network Architecture & Forward Pass

Let an input MNIST image ($28 \times 28$ pixels stored row-wise) be represented as a flattened row vector $\mathbf{x} \in \mathbb{R}^{1 \times 784}$. The network computes the forward pass in three primary stages:

1. **Hidden Layer:**

   $$\mathbf{z} = \mathbf{x} \mathbf{W}^{(1)} + \mathbf{b}^{(1)},$$

   where $\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times H}$ represents the first layer's weights, $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times H}$ is the bias, and $H \in \mathbb{N}$ is the number of hidden units.

2. **Non-linear Activation (ReLU):**

   $$\mathbf{z} = \max(0, \mathbf{z}),$$

   applied element-wise to introduce non-linearity, allowing the network to learn complex patterns.

3. **Output Layer (Logits):**

   $$\hat{\mathbf{y}} = \mathbf{z} \mathbf{W}^{(2)} + \mathbf{b}^{(2)},$$

   where $\mathbf{W}^{(2)} \in \mathbb{R}^{H \times 10}$ and $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times 10}$. The resulting vector $\hat{\mathbf{y}}$ contains the prediction scores for each of the 10 digit classes.

### Loss Function

To evaluate the network's predictions, this implementation uses the Mean Squared Error (MSE) loss.

Let $\mathbf{y} \in \mathbb{R}^{10}$ be the one-hot encoded ground-truth vector for the image, where the index corresponding to the true label is set to 1 and all other indices are 0.

The loss $\mathcal{L}$ for a single image is computed by taking the squared differences between the raw prediction scores (logits) $\hat{\mathbf{y}}$ and the target vector $\mathbf{y}$, and averaging them across all $C$ classes:

$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{C} \sum_{i=1}^{C} (\hat{y}_i - y_i)^2,$$

where $C = 10$ is the total number of digit classes. This scalar loss value is then added to the computation tape to seed the backward pass.

### Optimization & Tape-Based Autodiff

`zsl` utilizes a tape-based reverse-mode automatic differentiation engine. During the forward pass, every operation involving a `zsl.autodiff.Var(f32)` type is recorded sequentially onto a computation tape, constructing a Directed Acyclic Graph (DAG).

Once the scalar loss $\mathcal{L}$ is computed at the end of the forward pass, calling `backward()` on it traverses this tape in reverse. The engine applies the chain rule to calculate the gradient of the loss with respect to every parameter $\theta \in \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}\}$:

$$\nabla_{\theta} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}.$$

Once the gradients are accumulated, the network updates its parameters using standard Stochastic Gradient Descent (SGD) with a predefined learning rate $\eta$:

$$\theta_{t + 1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}.$$

### Batch Size

Because the `zsl` autodiff tape dynamically allocates memory to record operations, memory consumption scales linearly with the number of math operations executed before the tape is cleared.

If we were to process a mini-batch of 32 or 64 images, the tape would accumulate the operations of all those forward passes before a backward pass could be executed and the memory freed. To maintain a minimal memory footprint, this implementation processes a single image at a time (i.e., the batch size is 1). The loop follows this lifecycle per image:

1. Clear the tape.
2. Record the forward pass onto the tape.
3. Compute loss and traverse the tape backward.
4. Update weights via SGD.

## Prerequisites

- [Zig](https://ziglang.org/download/) (0.16.0)
- The MNIST dataset is already included in the `data/` directory, so no manual downloading is necessary.

## Building and Running

1. Clone this repository:

```bash
git clone https://github.com/srmadrid/zsl-mnist.git
cd zsl-mnist
```

2. Run the training loop. Compiling with the `ReleaseFast` optimization flag is highly recommended for reasonable execution times:

```bash
zig build run -Doptimize=ReleaseFast
```

## Results

When running for 5 epochs and with `seed = 42`, the following results are obtained:

```bash
  =======================================================
                    ZSL-MNIST TRAINING
  =======================================================
   EPOCH | LOSS       | TRAIN ACCURACY | TEST ACCURACY
  -------------------------------------------------------
     1   | 0.0271     |  87.88%        |  92.98%
     2   | 0.0150     |  94.26%        |  94.69%
     3   | 0.0126     |  95.65%        |  95.77%
     4   | 0.0112     |  96.39%        |  96.24%
     5   | 0.0103     |  96.79%        |  96.47%
  =======================================================
```
