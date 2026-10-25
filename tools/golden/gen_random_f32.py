"""Seeded random f32 input for the CI differential: gaussian blocks over six decades of scale, plus
exact half-way values. Finite, scales in [1e-4, 1e2] so neither side hits a reference-undefined case."""
import random, struct, sys

n_blocks = int(sys.argv[2])
rng = random.Random(int(sys.argv[3]))
xs = []
for _ in range(n_blocks):
    scale = 10 ** rng.uniform(-4, 2)
    block = [rng.gauss(0, 1) * scale for _ in range(32)]
    if rng.random() < 0.1:
        # Values on exact quantization ties for scale 1: amax 127 (q8_0) / 8 (q4_0).
        block[:8] = [127.0, 0.5, -0.5, 1.5, -1.5, 2.5, 8.0, -8.0]
    xs += block
open(sys.argv[1], "wb").write(struct.pack("<%df" % len(xs), *xs))
