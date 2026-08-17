# zsl-mnist

A custom neural network built from scratch for MNIST digit classification, designed to showcase the capabilities of the [zsl](https://github.com/srmadrid/zsl) library in Zig.

This repository trains a multi-layer perceptron using stochastic gradient descent. It relies entirely on `zsl` for the underlying mathematical operations, matrix manipulations, and backward-pass gradient calculations via its tape-based automatic differentiation engine.

> ⚠️ **Note on Performance:** This project is a proof-of-concept rather than a production-grade machine learning framework. It prioritizes demonstrating the mechanics of `zsl` over raw training speed.

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
   $$\hat{\mathbf{y}} = \mathbf{z} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$
   Where $\mathbf{W}^{(2)} \in \mathbb{R}^{H \times 10}$ and $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times 10}$. The resulting vector $\hat{\mathbf{y}}$ contains the prediction scores for each of the 10 digit classes.

### Optimization & Tape-Based Autodiff

`zsl` utilizes a tape-based reverse-mode automatic differentiation engine. During the forward pass, every operation involving a `zsl.autodiff.Var(f32)` type is recorded sequentially onto a computation tape, constructing a Directed Acyclic Graph (DAG).

Given a loss function $\mathcal{L}(\hat{\mathbf{y}}, y)$ representing the error between the prediction $\hat{\mathbf{y}}$ and the true label $y$, calling `backward()` traverses this tape in reverse. It applies the chain rule to compute the gradient of the loss with respect to every parameter $\theta \in \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}\}$:
$$\nabla_{\theta} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}$$
Once the gradients are accumulated, the network updates its parameters using standard Stochastic Gradient Descent (SGD) with a predefined learning rate $\eta$:
$$\theta_{t + 1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}$$

### Batch Size

Because the `zsl` autodiff tape dynamically allocates memory to record operations, memory consumption scales linearly with the number of math operations executed before the tape is cleared.

If we were to process a mini-batch of 32 or 64 images, the tape would accumulate the operations of all those forward passes before a backward pass could be executed and the memory freed. To maintain a minimal memory footprint, this implementation processes a single image at a time (batch size = 1). The loop follows this lifecycle per image:

1. Clear the tape and zero the parameter gradients.
2. Record the forward pass onto the tape.
3. Compute loss and traverse the tape backward.
4. Update weights via SGD.

## Prerequisites

- [Zig](https://ziglang.org/download/) (0.16.0)

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
