import random, struct, sys
# usage: gen_iq3_s_inputs.py in.f32 importance.f32   (importance holds one 512-element row; the input is whole 512-element rows)
# No exactly-zero 32-element sub-block is followed by a non-zero one: ggml 9d57ce456c mis-places the later sub-blocks
# (see iq3_s.rs), so those blocks are covered by a unit test, not by byte-identity.
random.seed(20261118)
xs = []
def g(scale, n=32):
    return [random.gauss(0, 1) * scale for _ in range(n)]
def sb(sub):
    out = []
    for s in sub:
        out += list(s) + [0.0] * (32 - len(s))
    return out
for b in range(60):
    s = 10 ** random.uniform(-4, 2)
    xs += [random.gauss(0, 1) * s for _ in range(256)]
for b in range(30):
    xs += sb([g(10 ** random.uniform(-3, 3)) for _ in range(8)])
# heavy tails and one-sided blocks
for b in range(10):
    xs += [abs(random.gauss(0, 1)) for _ in range(256)]
for b in range(10):
    xs += [random.choice([-1, 1]) * (10 ** random.uniform(-2, 2)) for _ in range(256)]
# sparse sub-blocks: a single non-zero element among exact zeros
for b in range(10):
    sub = []
    for i in range(8):
        v = [0.0] * 32
        v[random.randrange(32)] = random.gauss(0, 1) * 5
        sub.append(v)
    xs += sb(sub)
# one sub-block dominating, and exactly-zero sub-blocks only at the end
xs += sb([g(1000.0)] + [g(0.01) for _ in range(7)])
xs += sb([g(0.01) for _ in range(7)] + [g(1000.0)])
xs += sb([g(1.0), g(2.0), g(0.5), g(1.0), g(3.0), g(1.0), [], []])
xs += sb([g(1.0), g(2.0), g(0.5), g(1.0), g(3.0), g(1.0), g(0.1), [-0.0] * 32])
# all-zero super-block, negative-zero super-block
xs += [0.0] * 256
xs += [-0.0] * 256
# constant and alternating sign blocks (on-grid points), and exact level values 2l+1 at a unit scale
xs += sb([[0.3] * 32] * 8)
xs += sb([[1.0] * 32] * 8)
xs += sb([[1.0, -1.0] * 16] * 8)
xs += sb([[float(2 * (i % 8) + 1) for i in range(32)]] * 8)
xs += sb([[-float(2 * (i % 8) + 1) for i in range(32)]] * 8)
xs += sb([[float(2 * ((i * 3) % 8) + 1) for i in range(32)]] * 8)
# outliers: one large value among small ones
xs += [1.0e4] + [0.001 * (i + 1) for i in range(255)]
xs += [-1.0e4] + [0.001 * (i + 1) for i in range(255)]
xs += [0.5] * 255 + [-300.0]
# scale extremes: f16 overflow of d, tiny values
xs += [1.0e8] + [1.0e7 * (i + 1) for i in range(255)]
xs += [1.0e-10] + [2.0e-11 * i for i in range(255)]
xs += sb([g(1.0e6), g(1.0e-6)] + [g(1.0) for _ in range(6)])
while len(xs) % 512:
    xs += [random.gauss(0, 1) for _ in range(256)]
open(sys.argv[1], 'wb').write(struct.pack('<%df' % len(xs), *xs))
w = [random.uniform(0.0, 4.0) for _ in range(512)]
w[7] = 0.0
w[300] = 0.0
open(sys.argv[2], 'wb').write(struct.pack('<512f', *w))
print(len(xs), len(xs) // 256, len(xs) // 512)
