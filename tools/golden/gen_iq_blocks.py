"""Writes random raw IQ blocks (for the decode-only formats): every byte pseudo-random except the f16 scale, which is
kept finite and mid-range so the decoded values are ordinary floats.
usage: gen_iq_blocks.py out.bin <block_bytes> <nblocks> <seed>"""
import random, struct, sys
out, bb, nb, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
data = bytearray()
for _ in range(nb):
    blk = bytearray(rng.getrandbits(8) for _ in range(bb))
    exp = rng.randrange(8, 20)  # f16 exponent field: 2^-7 .. 2^4, never inf/nan/denormal
    h = (rng.getrandbits(1) << 15) | (exp << 10) | rng.getrandbits(10)
    blk[0:2] = struct.pack('<H', h)
    data += blk
open(out, 'wb').write(bytes(data))
