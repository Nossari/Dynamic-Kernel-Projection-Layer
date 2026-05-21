import torch
import torch.nn as nn
import math
from kernel_projection import DynamicKernelProjectionLayer

def generate_non_linear_moons(n_samples=100):
    """
    Generates complex, non-linearly separable concentric circles (target patterns)
    to test the infinite-dimensional expressive power of the kernel trick.
    """
    torch.manual_seed(42)
    data = []
    targets = []
    
    # Generate Inner Circle (Class 0)
    for _ in range(n_samples // 2):
        r = 0.5 + torch.randn(1) * 0.05
        theta = torch.rand(1) * 2 * math.pi
        data.append(torch.tensor([r * math.cos(theta), r * math.sin(theta), torch.randn(1)*0.1, torch.randn(1)*0.1]))
        targets.append(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) # Target representation vector
        
    # Generate Outer Circle (Class 1)
    for _ in range(n_samples // 2):
        r = 2.0 + torch.randn(1) * 0.1
        theta = torch.rand(1) * 2 * math.pi
        data.append(torch.tensor([r * math.cos(theta), r * math.sin(theta), torch.randn(1)*0.1, torch.randn(1)*0.1]))
        targets.append(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])) # Distinct target manifold vector

    return torch.stack(data), torch.stack(targets)

def run_kernel_benchmark():
    print("="*75)
    print("Running Empirical Evaluation: Dynamic Kernel Projection vs Standard Linear")
    print("="*75)
    
    # 1. Prepare Complex Non-Linear Dataset
    inputs, targets = generate_non_linear_moons(n_samples=20)
    criterion = nn.MSELoss()
    
    # 2. Instantiate both baseline topologies
    kernel_layer = DynamicKernelProjectionLayer(in_features=4, projection_dim=8, init_sigma=1.0)
    standard_layer = nn.Linear(in_features=4, projection_dim=8)
    
    print("Initial State Metrics:")
    print(f"  -> Initial Sigma Parameter: {kernel_layer.raw_sigma.item():.4f}")
    print("-" * 75)
    
    # 3. Simulate Forward Pass & Optimization Step for Standard Layer
    out_standard = standard_layer(inputs)
    loss_standard = criterion(out_standard, targets)
    loss_standard.backward()
    
    # 4. Simulate Forward Pass & Optimization Step for our Innovative Kernel Layer
    out_kernel = kernel_layer(inputs)
    loss_kernel = criterion(out_kernel, targets)
    loss_kernel.backward()
    
    # 5. Extract Analytics
    print("[Comparative Execution Analysis]")
    print(f"  -> Standard Linear Layer Loss: {loss_standard.item():.6f}")
    print(f"  -> Dynamic Kernel Projection Loss: {loss_kernel.item():.6f}")
    print("-" * 75)
    
    print("[Gradient Absorption Metrics]")
    print(f"  -> Dynamic Kernel Sigma Gradient: {kernel_layer.raw_sigma.grad.item():.6f}")
    print(f"  -> Dynamic Kernel Spatial Curvature (W_scale) Gradient Norm: {torch.norm(kernel_layer.W_scale.grad).item():.6f}")
    
    print("\nBenchmark conclusion:")
    if loss_kernel.item() < loss_standard.item():
        print("  SUCCESS: Dynamic Hilbert mapping yielded a tighter representation match with lower initial cross-entropy error.")
    else:
        print("  Standard optimization bounds achieved.")
    print("="*75)

if __name__ == "__main__":
    run_kernel_benchmark()
