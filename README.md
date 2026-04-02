# Collins

Adam's optimizer state takes 56 GB for a 7B model. Collins cuts that to 1.75 GB using Count-Sketch compression.

```bash
uv add git+https://github.com/lizixi-0x2F/Collins
```

---

I was training a 7B model and kept running out of memory. Not from the weights. Not from activations. From the optimizer.

Adam stores two momentum buffers per parameter: m and v. Both float32. For 7 billion parameters, that's 56 GB just for the optimizer state. Before you even load the model.

The standard fix is to use 8-bit optimizers or offload to CPU. Both slow things down. I wanted something that stayed fast.

---

Count-Sketch is a streaming algorithm from the 90s. It compresses data into fixed-size hash tables while keeping statistical properties intact.

Instead of storing full m and v vectors, Collins hashes each gradient element into a small table. A 7B model's optimizer state compresses from 56 GB to 1.75 GB. 32x smaller.

The hash tables are tiny enough to fit in GPU L1/L2 cache. This also removes the memory bandwidth bottleneck that slows down standard Adam at scale.

---

Here's how it works:

```
h(i) = ((a·i + b) mod P) mod K    # position hash
s(i) = {-1, +1}                    # sign hash
```

For the first moment m, gradients are sketched with a sign hash. This gives unbiased recovery: E[m̂] = m.

For the second moment v, gradients are sketched without the sign hash. Collisions overestimate v, which makes the denominator larger and updates more conservative. This acts like implicit regularization.

All EMA updates happen in compressed space. O(K) operations instead of O(N).

---

Usage:

```python
from collins import Collins

optimizer = Collins(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    compress_ratio=32,
    min_K=1024,
)
```

Same interface as torch.optim.Adam. No training loop changes.

---

Memory comparison:

| Model | Adam state | Collins state |
|---|---|---|
| 1B params | 8 GB | 250 MB |
| 7B params | 56 GB | 1.75 GB |
| 70B params | 448 GB | 14 GB |

---

Install:

```bash
uv add git+https://github.com/lizixi-0x2F/Collins
```

Or with pip:

```bash
pip install git+https://github.com/lizixi-0x2F/Collins
```

---

Limitations:

No sparse gradient support. Small parameters below 1024 elements don't compress. Best for large-scale training where optimizer memory is the bottleneck.

---

MIT License
