# Dynamic Kernel Projection Layer (DKPL) for Hilbert Space Mapping

This repository introduces a novel, high-performance architectural component designed to project linearly inseparable feature spaces into an infinite-dimensional Hilbert space approximation. Unlike traditional static kernel methods, the **Dynamic Kernel Projection Layer (DKPL)** features fully learnable spatial scales and bandwidths that dynamically optimize structural curvature during standard backpropagation.

---

## 🔬 Core Mathematical Architecture

Standard dense neural layers rely heavily on Euclidean linear transformations ($\mathbf{W} \cdot \mathbf{x} + \mathbf{b}$), requiring excessive depth to approximate non-linear boundaries. DKPL resolves this by mapping inputs into a highly expressive kernel space in a single operational step.

### 1. Parametric Spatial Scaling
Incoming feature matrices $\mathbf{X}$ are scaled along an empirical learnable spatial tensor $\mathbf{W}_{scale}$ to automatically adjust the sensitivity of individual feature components:
$$\mathbf{X}_{scaled} = \mathbf{X} \odot \mathbf{W}_{scale}$$

### 2. Pairwise Non-Euclidean Distance Calculation
Using the fundamental algebraic expansion property ($\|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\mathbf{a}\mathbf{b}^T$), the layer tracks localized distance variances cleanly inside eager execution:
$$D_{i,j} = \|\mathbf{x}_i\|_{scaled}^2 + \|\mathbf{x}_j\|_{scaled}^2 - 2\mathbf{x}_i \cdot \mathbf{x}_j^T$$

### 3. Learnable RBF Kernel Matrix & Hilbert Projection
The continuous dynamic similarity matrix is driven by a learnable bandwidth factor $\sigma$:
$$\mathbf{K}(i, j) = \exp\left( -\frac{D_{i,j}}{2\sigma^2} \right)$$

The final representation is fetched by projecting the non-linear self-similarity matrices into the target architectural dimensions:
$$\mathbf{O} = \mathbf{K}(\mathbf{X}, \mathbf{X}) \cdot (\mathbf{X} \cdot \mathbf{W}_{projection})$$

---

## 🚀 Key Advantages
* **Instant Non-Linear Separation:** Eradicates the necessity for deep network structures in highly complex classification domains (e.g., cybersecurity patterns, medical segmentations).
* **Learnable Manifold Variance:** Both the spatial curvature ($\mathbf{W}_{scale}$) and bandwidth ($\sigma$) dynamically absorb gradient updates.
* **Pure PyTorch Design:** Operates inside standard autograd loops without custom hardware-level dependency constraints.

---

## 📜 License
This project is open-sourced under the terms of the **MIT License**.
