"""Writes random raw IQ blocks (for the decode-only formats): every byte pseudo-random except the f16 scale, which is
kept finite and mid-range so the decoded values are ordinary floats.
usage: gen_iq_blocks.py out.bin <block_bytes> <nblocks> <seed> [iq1_m]
IQ1_M has no f16 field: its scale is spread over the top nibbles of four u16 in `scales` (block bytes 48..56); with
the `iq1_m` argument those nibbles are set to a finite mid-range f16 instead of bytes 0..2."""
import random, struct, sys
out, bb, nb, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
data = bytearray()
for _ in range(nb):
    blk = bytearray(rng.getrandbits(8) for _ in range(bb))
    exp = rng.randrange(8, 20)  # f16 exponent field: 2^-7 .. 2^4, never inf/nan/denormal
    h = (rng.getrandbits(1) << 15) | (exp << 10) | rng.getrandbits(10)
    if len(sys.argv) > 5 and sys.argv[5] == 'iq1_m':
        assert bb == 56
        for k in range(4):
            sc = struct.unpack_from('<H', blk, 48 + 2 * k)[0]
            struct.pack_into('<H', blk, 48 + 2 * k, (sc & 0x0fff) | (((h >> (4 * k)) & 0xf) << 12))
    else:
        blk[0:2] = struct.pack('<H', h)
    data += blk
open(out, 'wb').write(bytes(data))
