import random, struct, sys
# usage: gen_iq4_nl_inputs.py in.f32 importance.f32   (importance holds one 64-element row; the input is whole 64-element rows)
random.seed(20261116)
xs = []
def pad(vals):
    return list(vals) + [0.0] * (32 - len(vals))
# The first block must not be all zero: ggml leaves such a block's nibbles uninitialized when it is first.
xs += [random.gauss(0, 1) for _ in range(32)]
for b in range(95):
    s = 10 ** random.uniform(-4, 2)
    xs += [random.gauss(0, 1) * s for _ in range(32)]
# table end points and midpoints at scale 1: ties between neighbours, and the clamp at both ends
kv = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
xs += pad([-127.0] + [float(v) for v in kv])
xs += pad([-127.0] + [(a + b) / 2 for a, b in zip(kv, kv[1:])])
xs += pad([127.0] + [-float(v) for v in kv])
xs += pad([-127.0, 113.0])
# an all-zero block after non-zero ones (nibbles carried) and a negative-zero block
xs += pad([])
xs += [-0.0] * 32
# outliers: one large value among small ones, both signs
xs += [1.0e4] + [0.001 * (i + 1) for i in range(31)]
xs += [-1.0e4] + [0.001 * (i + 1) for i in range(31)]
xs += [0.5] * 31 + [-300.0]
# scale extremes: f16 overflow (inf scale), f16 subnormal and underflow, below GROUP_MAX_EPS
xs += [1.0e7] + [1.0e5 * (i + 1) for i in range(31)]
xs += [1.0e-14] + [-3.0e-15] * 31
xs += [1.0e-6] + [2.0e-7 * i for i in range(31)]
xs += [1.0e-16] * 32
xs += [0.3] * 32
xs += [1.0] * 32
# pad to whole 64-element rows
while len(xs) % 64:
    xs += [random.gauss(0, 1) for _ in range(32)]
open(sys.argv[1], 'wb').write(struct.pack('<%df' % len(xs), *xs))
w = [random.uniform(0.0, 4.0) for _ in range(64)]
w[7] = 0.0                                   # an element the importance vector switches off
open(sys.argv[2], 'wb').write(struct.pack('<64f', *w))
print(len(xs), len(xs) // 32, len(xs) // 64)
