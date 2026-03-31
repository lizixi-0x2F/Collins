# Hash-Adam (Count-Sketch Optimizer)

A memory-efficient Adam optimizer variant that compresses optimizer states (m, v) into compact hash tables via 2-Universal Hashing + Count-Sketch techniques.

## 📊 Key Features

- **Memory Efficiency**: O(K) per parameter instead of O(N), where K is the compressed hash table size
- **GPU Cache-Friendly**: Tiny hash tables fit in GPU L1/L2 cache, eliminating the memory wall bottleneck
- **Theoretically Grounded**: Count-Sketch with sign hashing for unbiased estimation with Chernoff-bound guarantees
- **Implicit Regularization**: Collision-based overestimation acts as natural regularization

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HashAdam Optimizer                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        ┌─────────────┐    ┌──────────────┐   ┌──────────────┐
        │  Parameter  │    │   Gradient   │   │ Compression  │
        │   Manager   │    │   Processor  │   │   Config     │
        └─────────────┘    └──────────────┘   └──────────────┘
                │                  │                  │
                └──────────────────┼─────────────────┘
                                   ▼
                        ┌────────────────────┐
                        │  State Initializer │
                        │  (Lazy Init)       │
                        └────────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        ┌─────────────┐    ┌──────────────┐   ┌──────────────┐
        │ Hash Seeds  │    │ Hash Func    │   │ Buffer Alloc │
        │ (a,b,c,d)   │    │ Manager      │   │ (m_t, v_t)   │
        └─────────────┘    └──────────────┘   └──────────────┘
                │                  │                  │
                └──────────────────┼─────────────────┘
                                   ▼
                        ┌────────────────────┐
                        │  Optimization Step │
                        └────────────────────┘
```

## 📈 Data Flow Diagram

```
                    Input Batch
                         │
                         ▼
                  ┌──────────────┐
                  │ Compute Loss │
                  └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │Backward Pass │
                  │(Gradients g) │
                  └──────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ For Each Parameter p │
              └──────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │Flatten  │    │ Check    │    │Lazy Init │
    │Gradient │    │ Grad Not │    │if needed │
    │         │    │ Sparse   │    │          │
    └─────────┘    └──────────┘    └──────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌──────────────────────┐
              │ Compute Hash Indices │
              │ h(i) = ((a*i+b)%P)%K │
              │ s(i) = {-1, +1}      │
              └──────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │Weight   │    │Count-    │    │Count-    │
    │Decay    │    │Sketch m_t│    │Sketch v_t│
    │(optional)   │(signed)  │    │(direct)  │
    └─────────┘    └──────────┘    └──────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌──────────────────────┐
              │ EMA Update (Biased)  │
              │ m_t ← β₁·m + (1-β₁)·m_hat │
              │ v_t ← β₂·v + (1-β₂)·v_hat │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Bias Correction      │
              │ bc1 = 1 - β₁^step    │
              │ bc2 = 1 - β₂^step    │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Reconstruct Moments  │
              │ m̂ = m_t[h_i] * s_i  │
              │ v̂ = v_t[h_i]        │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Compute Step Update  │
              │ Δp = -(lr/bc1)·m̂/(√v̂+ε)│
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Apply In-Place Update│
              │ p ← p + Δp           │
              └──────────────────────┘
                         │
                         ▼
                    Updated Weights
```

## 🔧 Implementation Details

### Core Components

| Component | Purpose | Complexity |
|-----------|---------|-----------|
| `__init__` | Initialize optimizer with hyperparameters | O(1) |
| `_init_state` | Lazy initialization of per-parameter hash tables | O(K) |
| `step` | Main optimization loop with sketching and update | O(N + K) |

### Hash Function Design

**2-Universal Hash Functions:**
- **Position Hash**: `h(i) = ((a*i + b) mod P) mod K`
  - Maps gradient indices to hash table slots
  - P = 2147483647 (Mersenne prime)
  
- **Sign Hash**: `s(i) = 2*((c*i + d) mod P mod 2) - 1`
  - Produces ±1 values for Count-Sketch
  - Ensures unbiased first-moment estimation

### Compression Parameters

- **compress_ratio** (default: 32): K = max(N // compress_ratio, min_K)
- **min_K** (default: 1024): Minimum hash table size for stability
- Optimal compression ratio ~34x for typical settings (ε=0.5, M=0.1, β₁=0.9)

## 🎯 Algorithm Summary

### First Moment (m) - Count-Sketch with Signing
```
sketched_m[h(i)] += s(i) * g_i  (unbiased via sign hashing)
m_t ← β₁·m_{t-1} + (1-β₁)·sketched_m
m̂_t ← m_t[h(i)] * s(i)  (recover unbiased estimate)
```

### Second Moment (v) - Direct Accumulation
```
sketched_v[h(i)] += g_i²  (no sign hash)
v_t ← β₂·v_{t-1} + (1-β₂)·sketched_v
v̂_t ← v_t[h(i)]  (direct lookup, always ≥ 0)
```

### Weight Update
```
step_size = lr / (1 - β₁^t)
p ← p - step_size * m̂_t / (√v̂_t + ε)
```

## 📦 Usage Example

```python
import torch
from torch_optim import Adam  # or your model
from Hash_Adam import HashAdam

# Create model
model = YourModel()

# Initialize HashAdam
optimizer = HashAdam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    compress_ratio=32,  # ~32x memory savings
    min_K=1024
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

## 🧮 Theoretical Guarantees

- **Unbiased First Moment**: Count-Sketch with sign hashing provides unbiased estimation of m_t
- **Chernoff Bound**: Error probability bounded exponentially with compression ratio
- **Safe Second Moment**: Direct accumulation ensures v_t never goes negative
- **Implicit Regularization**: Collision-based noise acts as natural L2 regularization

## 💾 Memory Complexity

| Baseline Adam | HashAdam |
|---|---|
| 2N × 32-bit per param | 2K × 32-bit per param |
| Example: 1B params = 8GB | Example: 1B params ≈ 250MB |

**Speedup**: Gradient descent step size reduced from O(N) to O(K), better cache locality.

## ⚠️ Limitations

- Does not support sparse gradients (PyTorch enforces dense grad computation)
- Requires careful tuning of `compress_ratio` for different problem sizes
- Best suited for large parameter count scenarios where memory dominates

## 📄 License

Open source - academic use

## 👨‍💻 Author

lizixi-0x2F
