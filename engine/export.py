"""Integer parameters -> one binary blob (the cartridge's data) + a C header of offsets.

Ternary matrices are stored as the lookup kernel reads them: one byte per 4 weights, and the byte is a
pointer step between 81-entry tables (see bench/tern.s)."""
import sys, pickle, struct, numpy as np
from ref_int import quantize, EXP_N, EXP_BITS, Q, EMA_K


def steps(trits):
    """(rows, n_in) trits -> (rows, n_in/4) step bytes. Table entry index = a + 3b + 9c + 27d with codes 0:0, 1:+, 2:-."""
    t = np.asarray(trits)
    code = np.where(t > 0, 1, np.where(t < 0, 2, 0)).reshape(t.shape[0], -1, 4)
    idx = code[..., 0] + 3 * code[..., 1] + 9 * code[..., 2] + 27 * code[..., 3]
    st = np.empty_like(idx)
    st[:, 0] = idx[:, 0] + 1
    st[:, 1:] = 81 + idx[:, 1:] - idx[:, :-1]
    assert st.min() >= 1 and st.max() <= 255
    return st.astype(np.uint8)


class Blob:
    def __init__(self):
        self.b, self.defs = bytearray(), []

    def add(self, name, a, dtype):
        while len(self.b) % 4: self.b.append(0)
        a = np.ascontiguousarray(np.asarray(a).astype(dtype))
        self.defs.append((name, len(self.b)))
        self.b += a.tobytes()

    def const(self, name, v):
        self.defs.append((name, int(v)))


def export(I, out_bin, out_h, extra=None):
    c = I["cfg"]
    B = Blob()
    for k in ("V", "D", "L", "T", "NH", "DK", "DV", "HD", "E"):
        B.const("K_" + k, c[k])
    B.const("K_TOPK", c["topk"]); B.const("K_TOPN", 40); B.const("K_Q", Q); B.const("K_EXP_N", EXP_N); B.const("K_EXP_BITS", EXP_BITS)
    B.const("K_NATT", len(I["att_layers"])); B.const("K_ATT_MASK", sum(1 << l for l in I["att_layers"]))
    em = I["emb"] & 63                                                # 6-bit two's complement, 4 values in 3 bytes
    assert int(np.abs(I["emb"]).max()) <= 31 and I["emb_step"].max() < 65536
    u = em[:, 0::4] | (em[:, 1::4] << 6) | (em[:, 2::4] << 12) | (em[:, 3::4] << 18)
    B.add("O_EMB", np.stack([u & 255, (u >> 8) & 255, (u >> 16) & 255], -1), np.uint8)
    B.add("O_EMB_STEP", I["emb_step"], np.uint16)
    B.add("O_ROPE_INC", I["rope_inc"], np.uint32); B.add("O_SIN", I["sin"], np.int16); B.add("O_EXP", I["exp"], np.uint16)
    lay = []
    for l, J in enumerate(I["layers"]):
        o = {}
        def add(nm, a, dt):
            B.add(f"O_L{l}_{nm}", a, dt); o[nm] = B.defs[-1][1]
        add("N1", J["n1"], np.int32); add("N2", J["n2"], np.int32)
        if l in I["att_layers"]:
            add("Q", steps(J["q"]), np.uint8); add("K", steps(J["k"]), np.uint8); add("V", steps(J["v"]), np.uint8)
            add("O", steps(J["o"]), np.uint8); add("KG", J["k_g"], np.int32)
            add("ATT", [J["v_m"], J["o_m"], *J["c"]], np.int32); add("DBIAS", J["dbias"], np.int32)
        else:
            add("MIXC", np.concatenate([J["taps"], J["avg"]]).T, np.int32)      # per dimension: 3 taps, 3 average weights
        add("W1", steps(J["w1"].reshape(-1, c["D"])), np.uint8); add("W2", steps(J["w2"].reshape(-1, c["HD"])), np.uint8)
        add("W1M", J["w1_m"], np.int32); add("W2M", J["w2_m"], np.int32); add("ROUTE", J["route"], np.uint8)
        lay.append(o)
    for n, sp in enumerate(I["inj"]):
        assert sp["layer"] == 0
        lev = sp["t"] & 15                                           # two's-complement nibbles, low nibble first
        B.add(f"O_INJ{n}", lev[:, 0::2] | (lev[:, 1::2] << 4), np.uint8)
        B.add(f"O_INJ{n}_STEP", sp["step"], np.uint16)
        B.const(f"K_INJ{n}_ROWS", sp["rows"]); B.const(f"K_INJ{n}_SALT", sp["salt"]); B.const(f"K_INJ{n}_TRI", int(sp["kind"] == "trigram"))
    B.const("K_NINJ", len(I["inj"]))
    B.add("O_NF", I["nf"], np.int32); B.add("O_HEAD", steps(I["head"]), np.uint8); B.add("O_HEAD_M", I["head_m"], np.uint16)
    B.const("K_COPY_GAIN", I["copy_gain"])
    if extra is not None:                                             # fast output step (engine/outstep.py)
        O = extra
        B.add("O_SHORT", O["short"], np.uint16); B.add("O_SUCC", O["succ"], np.uint16)
        B.add("O_ROUTER", steps(O["router"]), np.uint8); B.add("O_ROUTER_M", O["router_m"], np.int32)
        B.add("O_GSTART", O["group_start"], np.uint16); B.add("O_GMEM", O["group_members"], np.uint16)
        B.const("K_SHORT", len(O["short"])); B.const("K_KSUCC", O["succ"].shape[1]); B.const("K_NGROUP", len(O["router"]))
    from tok_export import tables
    TK = tables()
    B.add("O_TOK_BYTE", TK["byte"], np.uint16); B.add("O_TOK_MKEY", TK["mkey"], np.uint32); B.add("O_TOK_MVAL", TK["mval"], np.uint32)
    B.add("O_TOK_OFF", TK["off"], np.uint32); B.add("O_TOK_TEXT", TK["text"], np.uint8); B.const("K_NMERGE", len(TK["mkey"]))
    from words import flags
    wsp, wst, _ = flags()
    pack = lambda m: np.packbits(m.astype(np.uint8), bitorder="little").view(np.uint32)
    B.add("O_WORD_SP", pack(wsp), np.uint32); B.add("O_WORD_START", pack(wst), np.uint32)
    FT = pickle.load(open("/home/user/kokosd/engine/facts_table.pkl", "rb"))
    B.add("O_FACT_HASH", FT["hash"], np.uint32); B.add("O_FACT_OFF", FT["off"], np.uint32); B.add("O_FACT_TEXT", FT["text"], np.uint8)
    B.const("K_NFACT", len(FT["hash"]))
    # per-layer offset table so the C code can loop over layers
    names = ["N1", "N2", "Q", "K", "V", "O", "KG", "ATT", "DBIAS", "MIXC", "W1", "W2", "W1M", "W2M", "ROUTE"]
    tab = [[lay[l].get(nm, 0) for nm in names] for l in range(c["L"])]
    B.add("O_LAYERS", tab, np.uint32)
    while len(B.b) % 4: B.b.append(0)
    open(out_bin, "wb").write(B.b)
    with open(out_h, "w") as f:
        f.write("/* generated by engine/export.py */\n#ifndef KOKO_LAYOUT_H\n#define KOKO_LAYOUT_H\n")
        for nm, v in B.defs: f.write(f"#define {nm} {v}u\n" if nm.startswith("O_") else f"#define {nm} {v}\n")
        f.write("enum { " + ", ".join("LO_" + n for n in names) + ", LO_N };\n")
        f.write(f"#define KOKO_BLOB_BYTES {len(B.b)}u\n#endif\n")
    return len(B.b)


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "i_kd"
    I = quantize(pickle.load(open(f"/home/user/kokosd/engine/params_{run}.pkl", "rb")))
    O = pickle.load(open(f"/home/user/kokosd/engine/outstep_{run}.pkl", "rb"))
    n = export(I, "/home/user/kokosd/build/koko.bin", "/home/user/kokosd/gen/koko_layout.h", O)
    print("blob", n, "bytes =", round(n / 2 ** 20, 2), "MB")
