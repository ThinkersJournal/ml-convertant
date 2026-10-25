import random, struct, sys
random.seed(20261025)
xs = []
# 64 blocks of gaussian data at assorted scales
for b in range(64):
    s = 10 ** random.uniform(-4, 2)
    xs += [random.gauss(0, 1) * s for _ in range(32)]
def block(vals):
    assert len(vals) == 32
    return list(vals)
# zeros, negative zero
xs += block([0.0] * 32)
xs += block([-0.0] * 32)
# ties: amax 127 -> d = 1, values land on .5
xs += block([127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 126.5, -126.5, 3.5] + [0.0] * 22)
# one huge outlier among small values
xs += block([1.0e4] + [0.001 * (i + 1) for i in range(31)])
# f16 scale overflow (d > 65504): reference yields inf scale
xs += block([1.0e30] + [1.0] * 31)
# f16 scale underflow (d < 2^-25): dequantizes to 0
xs += block([1.0e-8] + [-1.0e-9] * 31)
# f16 subnormal scale
xs += block([127 * 3.0e-6] + [1.0e-6 * i for i in range(31)])
# single nonzero
xs += block([0.0] * 31 + [-3.25])
# all equal
xs += block([0.3] * 32)
assert len(xs) % 32 == 0
open(sys.argv[1], 'wb').write(struct.pack('<%df' % len(xs), *xs))
print(len(xs), len(xs) // 32)
