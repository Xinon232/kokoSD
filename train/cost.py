"""Estimated GBA cost of a Config, from the constants measured in research/bench-results.md.

Rough by design (within about 10% of the benchmarked 256-wide shapes): it exists so that every
experiment reports quality next to an estimate of cartridge speed.
"""
from __future__ import annotations

CPS = 16_777_216


def _rows(n_rows, n_in, cfg):
    """Cycles for n_rows matrix rows of n_in inputs."""
    if cfg.wq == "int8" or cfg.wq == "fp":       # float models would ship as 8-bit
        return n_rows * (30 + 7.5 * n_in)
    return n_rows * (25 + 27.5 * n_in / 16)


def _build(n_in, cfg):
    if cfg.wq in ("int8", "fp"):
        return 0
    return n_in / 4 * (110 if cfg.wq == "sherry" else 270)


def estimate(cfg, n_cls=None):
    c = cfg
    norm = 49 * c.D
    hk = c.DK // c.NH
    if c.keyq == "none":                          # 8-bit keys: a multiply per key dim per position
        scan = c.NH * c.T * hk * 7.5
    else:                                         # 1-bit stored keys: one lookup per 4 key dims (42 cycles measured at hk=16)
        scan = c.NH * c.T * (15 + 27.5 * hk / 16)
        if c.keyq == "pq16":                      # per-query codeword tables built with multiplies (not benchmarked)
            scan += c.NH * (hk // 4) * 16 * 4 * 7.5
    att = (norm + _build(c.D, c) + _rows(2 * c.DK + c.DV, c.D, c)
           + _build(c.DK, c) + scan + c.NH * 7300
           + _build(c.DV, c) + _rows(c.D, c.DV, c))
    mlp = norm + _build(c.D, c) + _rows(c.HD, c.D, c) + _build(c.HD, c) + _rows(c.D, c.HD, c)
    att_w = (2 * c.DK + c.DV) * c.D + c.DV * c.D
    n_att = c.L if c.att_layers is None else len(c.att_layers)
    active = 2 * c.HD * c.D
    stored = active
    if c.mlp == "moe":
        if c.router in ("learned", "frozen"):
            mlp += _rows(c.E, c.D, c) + 500
            active += c.E * c.D
            stored += c.E * c.D
        stored += (c.E - 1) * 2 * c.HD * c.D
    if c.keyq == "rescore":                       # exact rescoring of the shortlisted positions with multiplies
        att += c.NH * c.rescore_m * (c.DK // c.NH) * 7.5
    if c.att_bias:
        att += c.NH * c.T * 6
    mixer = norm + 6 * c.D * 12 if c.mixer == "diag" else 0
    hm = len(c.hashmem_layers) * (_rows(c.hashmem_heads * c.hashmem_bits, c.D, c) + c.hashmem_heads * c.D * 6) \
        if c.hashmem_heads else 0
    inj = sum(len(t["layers"]) * (4000 if t.get("gate") else 1500) for t in c.inject)
    # output head: cost of the head this arm trains, plus what a two-step head would cost for the same model
    n_cls = n_cls or max(1, round(c.V ** 0.5))
    two_rows = n_cls + c.V // n_cls
    if c.head == "two_level":
        n_tail = c.V - c.head_short
        C = c.head_C or max(1, round(n_tail ** 0.5))
        two_rows = c.head_short + C + -(-n_tail // C)
    head_rows = two_rows if c.head == "two_level" else c.V
    body = n_att * att + (c.L - n_att) * mixer + c.L * mlp + inj + hm + 5400
    head = lambda r: norm + _build(c.D, c) + _rows(r, c.D, c)
    cycles, cycles_two = body + head(head_rows), body + head(two_rows)
    table_rows = c.V + sum(t["rows"] * len(t["layers"]) * (2 if t.get("gate") else 1)
                           * (t.get("bits") or 8) / 8 for t in c.inject)
    table_rows += len(c.hashmem_layers) * c.hashmem_heads * 2 ** c.hashmem_bits
    bits = 8 if c.wq in ("int8", "fp") else 2
    rom = (n_att * att_w + c.L * stored + c.V * c.D) * bits / 8 + table_rows * c.D
    return {"cycles_per_token": round(cycles), "gba_tokens_per_s": round(CPS / cycles, 2),
            "gba_tokens_per_s_two_step_head": round(CPS / cycles_two, 2),
            "active_weights": n_att * att_w + c.L * active + head_rows * c.D,
            "stored_weights": n_att * att_w + c.L * stored + c.V * c.D, "rom_mb": round(rom / 2 ** 20, 2)}
