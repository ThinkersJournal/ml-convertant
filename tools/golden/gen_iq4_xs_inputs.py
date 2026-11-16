import random, struct, sys
# usage: gen_iq4_xs_inputs.py in.f32 importance.f32   (importance holds one 512-element row; the input is whole 512-element rows)
random.seed(20261117)
xs = []
def sb(sub):
    """a 256-element super-block from eight 32-element sub-blocks (lists)"""
    out = []
    for s in sub:
        out += list(s) + [0.0] * (32 - len(s))
    return out
def g(scale, n=32):
    return [random.gauss(0, 1) * scale for _ in range(n)]
# plain random super-blocks at a spread of scales
for b in range(40):
    s = 10 ** random.uniform(-4, 2)
    xs += [random.gauss(0, 1) * s for _ in range(256)]
# sub-blocks of very different magnitude (scales spread across the 6-bit range)
for b in range(20):
    xs += sb([g(10 ** random.uniform(-3, 3)) for _ in range(8)])
# sign mix: sub-blocks whose largest element is positive and negative (scales of both signs, clamp at 31)
for b in range(10):
    sub = []
    for i in range(8):
        v = g(1.0)
        v[random.randrange(32)] = (4.0 if i % 2 else -4.0) * (1 + i)
        sub.append(v)
    xs += sb(sub)
# one sub-block dominating the rest
xs += sb([g(1000.0)] + [g(0.01) for _ in range(7)])
xs += sb([g(0.01) for _ in range(7)] + [g(1000.0)])
# all-zero and below-threshold sub-blocks inside a non-zero super-block
xs += sb([g(1.0), [], [0.0] * 32, [1.0e-16] * 32, g(2.0), [-0.0] * 32, g(0.5), [3.0e-15] * 32])
# all-zero super-block, negative-zero super-block, below-threshold super-block
xs += [0.0] * 256
xs += [-0.0] * 256
xs += [1.0e-16] * 256
# identical sub-blocks; constant blocks
xs += sb([[0.3] * 32] * 8)
xs += sb([[1.0] * 32] * 8)
xs += sb([[-1.0] * 32] + [[1.0] * 32] * 7)
# table end points and midpoints at scale 1 in several sub-blocks (ties, clamps)
kv = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
t = [-127.0] + [float(v) for v in kv]
m = [-127.0] + [(a + b) / 2 for a, b in zip(kv, kv[1:])]
xs += sb([t, m, [127.0] + [-float(v) for v in kv], [-127.0, 113.0], t, m, [127.0] + [-float(v) for v in kv], [-127.0, 113.0]])
# half-way sub-block scales: scale 64 sets d = -2, scales 33 and 35 give l = -16.5 and -17.5 (ties to even)
xs += sb([[-127.0 * a] + [0.0] * 15 + [113.0 * a] for a in (64.0, 33.0, 35.0, 37.0, 63.0, 1.0, 2.0, 3.0)])
# outliers: one large value among small ones
xs += [1.0e4] + [0.001 * (i + 1) for i in range(255)]
xs += [-1.0e4] + [0.001 * (i + 1) for i in range(255)]
xs += [0.5] * 255 + [-300.0]
# scale extremes: f16 overflow of d, f16 subnormal/underflow of d, a tiny sub-block next to a huge one
xs += [1.0e7] + [1.0e5 * (i + 1) for i in range(255)]
xs += [1.0e10] + [1.0e9 * (i + 1) for i in range(255)]   # d = max_scale / 32 exceeds f16's range
xs += [1.0e-6] + [2.0e-7 * i for i in range(255)]
xs += [1.0e-14] + [-3.0e-15] * 255
xs += sb([g(1.0e6), g(1.0e-6)] + [g(1.0) for _ in range(6)])
# the first super-block must not be special: pad to whole 512-element rows with random data
while len(xs) % 512:
    xs += [random.gauss(0, 1) for _ in range(256)]
open(sys.argv[1], 'wb').write(struct.pack('<%df' % len(xs), *xs))
w = [random.uniform(0.0, 4.0) for _ in range(512)]
w[7] = 0.0
w[300] = 0.0
open(sys.argv[2], 'wb').write(struct.pack('<512f', *w))
print(len(xs), len(xs) // 256, len(xs) // 512)
