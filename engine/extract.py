"""Pull a trained model apart into plain numpy arrays: trits, scales, gains, tables. No torch after this."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import sys, numpy as np, torch
sys.path.insert(0, ROOT + '/study'); sys.path.insert(0, ROOT)
from common import load


def trits(w):
    """(..., out, in) latent weights -> int8 trits and one float scale per matrix.

    Uses the training code's own float32 quantiser: many latent weights sit within rounding error of
    the +-0.5 threshold, so recomputing the rule in float64 flips some of them."""
    from train import quant
    wq = quant.tern_values(w)
    s = wq.abs().flatten(-2).amax(-1)
    return torch.sign(wq).to(torch.int8).numpy(), s.double().numpy()


def rows4(v, bits=4):
    """Lookup rows stored at 4 bits: integer levels -7..7 and one float step per row (training code's rounding)."""
    from train import quant
    vq = quant.act_values(v, bits)
    q = float(2 ** (bits - 1) - 1)
    step = (vq.abs().amax(-1) / q).double()
    lev = torch.round(vq.double() / step.clamp_min(1e-30)[:, None]).to(torch.int8)
    return lev.numpy(), step.numpy()


def extract(run="i_kd"):
    model, cfg, counts, res = load(run)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    P = {"cfg": {k: getattr(cfg, k) for k in ("V", "D", "L", "T", "NH", "DK", "DV", "HD", "E", "topk")},
         "att_layers": list(cfg.att_layers), "emb": sd["emb.weight"].double().numpy(), "layers": []}
    for l in range(cfg.L):
        Ly = {"n1": sd[f"n1.{l}.g"].double().numpy(), "n2": sd[f"n2.{l}.g"].double().numpy()}
        if cfg.has_att(l):
            for nm in "qkvo":
                Ly[nm], Ly[nm + "_s"] = trits(sd[f"att.{l}.{nm}.weight"])
            Ly["k_g"] = sd[f"att.{l}.k.g"].double().numpy()
            Ly["temp"] = sd[f"att.{l}.temp"].double().clamp(-3, 3).numpy()
            Ly["dbias"] = sd[f"att.{l}.dbias"].double().numpy()
        else:
            Ly["taps"] = sd[f"att.{l}.taps"].double().numpy(); Ly["avg"] = sd[f"att.{l}.avg"].double().numpy()
        Ly["w1"], Ly["w1_s"] = trits(sd[f"mlp.{l}.w1.weight"])
        Ly["w2"], Ly["w2_s"] = trits(sd[f"mlp.{l}.w2.weight"])
        Ly["route"] = sd[f"mlp.{l}.route"].numpy().astype(np.int64)
        P["layers"].append(Ly)
    P["inj"] = []
    for key, m in model.inj.items():
        l, n = key.split("_")
        t, step = rows4(sd[f"inj.{key}.v.weight"], m.bits)
        P["inj"].append({"layer": int(l), "kind": m.kind, "rows": m.rows, "salt": m.salt, "t": t, "step": step})
    P["nf"] = sd["nf.g"].double().numpy()
    P["head"], P["head_s"] = trits(sd["head.weight"])
    P["head_g"] = sd["head.g"].double().numpy()
    P["copy_gain"] = float(sd["copy_gain"])
    return P, model, cfg


if __name__ == "__main__":
    import pickle
    P, _, _ = extract(sys.argv[1] if len(sys.argv) > 1 else "i_kd")
    pickle.dump(P, open(f"{ROOT}/engine/params_{sys.argv[1] if len(sys.argv) > 1 else 'i_kd'}.pkl", "wb"))
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in P.items() if k not in ("layers", "inj")})
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in P["layers"][1].items()})
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in P["layers"][0].items()})
