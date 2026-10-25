import random, struct, sys
random.seed(20261026)
xs = []
for b in range(64):
    s = 10 ** random.uniform(-4, 2)
    xs += [random.gauss(0, 1) * s for _ in range(32)]
def pad(vals):
    return list(vals) + [0.0] * (32 - len(vals))
xs += pad([])                                   # zeros
xs += [-0.0] * 32                               # negative zeros
# max=-8 (d=1, id=1): halves are tie points; +8 clips to 15
xs += pad([-8.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 7.5, -7.5, 8.0])
# max=+8 (d=-1, id=-1): tie points flip sides
xs += pad([8.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 7.5, -7.5, -8.0])
# equal |max|, first wins (-8 before 8) and (8 before -8)
xs += pad([-8.0, 8.0, 3.0])
xs += pad([8.0, -8.0, 3.0])
# outlier among small values
xs += [1.0e4] + [0.001 * (i + 1) for i in range(31)]
# f16 scale overflow / underflow / subnormal
xs += [1.0e30] + [1.0] * 31
xs += [1.0e-8] + [-1.0e-9] * 31
xs += [8 * 3.0e-6] + [1.0e-6 * i for i in range(31)]
xs += pad([0.0] * 31 + [-3.25])
xs += [0.3] * 32
# tiny max whose f32 scale rounds to zero (id = 0)
xs += pad([1.0e-45])
assert len(xs) % 32 == 0
open(sys.argv[1], 'wb').write(struct.pack('<%df' % len(xs), *xs))
print(len(xs), len(xs) // 32)
