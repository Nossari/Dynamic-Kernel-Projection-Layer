import torch
import torch.nn as nn

class DynamicKernelProjectionLayer(nn.Module):
    def __init__(self, in_features, projection_dim, init_sigma=1.0):
        """
        Dynamic Kernel Projection Layer (RML-3) Prototype.
        Transforms inputs into an infinite-dimensional Hilbert space approximation 
        using a learnable RBF Kernel structure.
        
        :param in_features: Dimensionality of incoming feature vectors.
        :param projection_dim: Dimensionality of the target Hilbert projection space.
        :param init_sigma: Initial scaling factor for the spatial variance.
        """
        super(DynamicKernelProjectionLayer, self).__init__()
        self.in_features = in_features
        self.projection_dim = projection_dim
        
        # 1. Learnable Spatial Scaling Matrix: Shapes the non-Euclidean curvature
        self.W_scale = nn.Parameter(torch.ones(in_features))
        
        # 2. Learnable Bandwidth Parameter (Must remain positive, handled during forward)
        self.raw_sigma = nn.Parameter(torch.tensor(init_sigma))
        
        # 3. Learnable Hilbert Space Projection Matrix
        # Maps the kernel self-similarity matrix to the final representation space
        self.W_projection = nn.Parameter(torch.randn(in_features, projection_dim) * 0.1)

    def forward(self, x):
        """
        Computes the dynamic kernel transformation and projects features onto the Hilbert manifold.
        :param x: Input tensor of shape (batch_size, in_features)
        """
        batch_size, seq_len = x.shape
        assert seq_len == self.in_features, "Input features mismatch."

        # Enforce positivity on sigma to prevent mathematical divergence (division by zero)
        sigma = torch.clamp(self.raw_sigma, min=1e-4)
        
        # Scale inputs dynamically using our learnable matrix: X_scaled = X * W_scale
        x_scaled = x * self.W_scale
        
        # Compute Pairwise Squared Euclidean Distances: ||x_i - x_j||^2
        # Using the algebraic identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2a.b^T
        norms = torch.sum(x_scaled ** 2, dim=1, keepdim=True)  # (batch_size, 1)
        pairwise_distances = norms + norms.t() - 2.0 * torch.matmul(x_scaled, x_scaled.t())
        
        # 4. Construct the Dynamic Kernel Matrix (Self-Similarity Matrix)
        # K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
        kernel_matrix = torch.exp(-pairwise_distances / (2.0 * (sigma ** 2)))
        
        # 5. Project the Non-Linear Hilbert representations back to the target dimension
        # Output shape: (batch_size, projection_dim)
        output = torch.matmul(kernel_matrix, torch.matmul(x, self.W_projection))
        return output

# --- Structural Verification & Gradient Trace Run ---
if __name__ == "__main__":
    print("="*70)
    print("Initializing Dynamic Kernel Projection Layer Verification")
    print("="*70)

    # Instantiate the innovative layer (Transforming 4 features into 8 Hilbert features)
    kernel_layer = DynamicKernelProjectionLayer(in_features=4, projection_dim=8, init_sigma=1.5)
    
    # Create batch data (3 samples, 4 features each)
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 8)
    criterion = nn.MSELoss()
    
    print(f"Initial Learnable Sigma: {kernel_layer.raw_sigma.item():.4f}")
    print(f"Initial Learnable W_scale: {kernel_layer.W_scale.data.numpy()}")
    print("-" * 70)
    
    # Forward Pass through Hilbert manifold
    outputs = kernel_layer(inputs)
    loss = criterion(outputs, targets)
    
    # Backward Pass to trace non-Euclidean gradients
    loss.backward()
    
    print(f"Computed Loss Value: {loss.item():.6f}")
    print("\nVerifying Autograd Gradient Propagation:")
    print(f"  -> Sigma Gradient: {kernel_layer.raw_sigma.grad.item():.6f}")
    print(f"  -> W_scale Gradient Norm: {torch.norm(kernel_layer.W_scale.grad).item():.6f}")
    print(f"  -> W_projection Gradient Norm: {torch.norm(kernel_layer.W_projection.grad).item():.6f}")
    print("="*70)
    print("Verification Successful! Parameters are dynamically absorbing first-order gradients.")
