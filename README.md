Hash-Adam (Count-Sketch Optimizer)
==================================
A memory-efficient Adam variant that compresses optimizer states (m, v)
into compact hash tables via 2-Universal Hashing + Count-Sketch.

Optimizer state memory: O(K) per parameter instead of O(N).
The tiny hash tables can fit in GPU L1/L2 cache, smashing the memory wall.

Theory: First moment (m) uses Count-Sketch with sign hashing s(i) ∈ {-1, +1}
        for unbiased estimation with Chernoff-bound guarantees.
        Second moment (v) uses direct hash accumulation (no sign hash) —
        collisions safely overestimate v, acting as implicit regularization.
