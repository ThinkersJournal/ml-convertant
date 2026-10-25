import random, struct, sys
random.seed(20261032)
xs = []
def gauss_block(scale, offset=0.0):
    return [random.gauss(0, 1) * scale + offset for _ in range(256)]
for b in range(48):
    xs += gauss_block(10 ** random.uniform(-4, 2), random.choice([0.0, 0.0, 1.0, -1.0]) * 10 ** random.uniform(-3, 1))
xs += [0.0] * 256                                   # zeros
xs += [0.5] * 256                                   # constant positive (max == min after min clamp? no: max != min)
xs += [-0.5] * 256                                  # constant negative
xs += [3.0 + 0.01 * (i % 7) for i in range(256)]    # all positive: min clamped to 0
xs += [-3.0 - 0.01 * (i % 7) for i in range(256)]   # all negative
b = gauss_block(0.1); b[17] = 40.0; xs += b          # one outlier
b = gauss_block(0.1); b[200] = -40.0; xs += b
# sub-blocks of very different magnitude inside one super-block
b = []
for j in range(8):
    b += gauss_block(10 ** (j - 4))[:32]
xs += b
# scales near f16 range limits (scale/63 near subnormal; large values)
xs += gauss_block(1e-6)
xs += gauss_block(3e3)
# integer ramp (exact levels), and a block with one nonzero
xs += [float(i % 16) for i in range(256)]
b = [0.0] * 256; b[255] = -2.5; xs += b
assert len(xs) % 256 == 0
open(sys.argv[1], 'wb').write(struct.pack('<%df' % len(xs), *xs))
print(len(xs), len(xs) // 256)
