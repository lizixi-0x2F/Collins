"""
Collins: Memory-efficient Adam optimizer via Count-Sketch compression.

Reduces optimizer state memory from O(N) to O(K) per parameter,
where K = N / compress_ratio. Hash tables fit in GPU L1/L2 cache.
"""

from .optimizer import Collins

__version__ = "0.1.0"
__author__ = "lizixi-0x2F"
__all__ = ["Collins"]
