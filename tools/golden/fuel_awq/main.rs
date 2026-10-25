#[path = "awq_repack.rs"]
mod awq_repack;
use std::io::Write;
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let mut f = std::fs::File::create(out).unwrap();
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let cases = [(256usize, 64usize, 128usize), (128, 128, 64), (1024, 64, 64), (128 * 9, 16, 128), (512, 192, 128)];
    f.write_all(&(cases.len() as u32).to_le_bytes()).unwrap();
    for (ic, oc, g) in cases {
        let groups = ic / g;
        let qw: Vec<u32> = (0..ic * oc / 8).map(|_| next() as u32).collect();
        let qz: Vec<u32> = (0..groups * oc / 8).map(|_| next() as u32).collect();
        let sc: Vec<half::f16> = (0..groups * oc).map(|_| half::f16::from_bits(0x2000 + (next() % 0x3000) as u16)).collect();
        let k = awq_repack::repack_hf_awq_for_kernel(&qw, &qz, &sc, ic, oc, g).unwrap();
        for v in [ic, oc, g, k.zeros_w] { f.write_all(&(v as u32).to_le_bytes()).unwrap(); }
        for v in [qw.len(), qz.len(), sc.len(), k.weight.len(), k.zeros.len(), k.scales.len()] { f.write_all(&(v as u32).to_le_bytes()).unwrap(); }
        for w in qw.iter().chain(&qz).chain(&k.weight).chain(&k.zeros) { f.write_all(&w.to_le_bytes()).unwrap(); }
        for h in sc.iter().chain(&k.scales) { f.write_all(&h.to_bits().to_le_bytes()).unwrap(); }
    }
}
