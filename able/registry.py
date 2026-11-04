"""Which checkpoint goes with which config.

This exists because the pairing was not written down anywhere. A checkpoint under
`morphomnist/ladder/k5_s42/` turned out to be reproduced by `adv_k5_s42.yaml`, and nothing in the
filename or the folder said so -- the only way to tell was to build each candidate config and
compare state_dict keys. Getting it wrong does not raise: PyTorch happily reports missing and
unexpected keys and carries on with a partly random model.

So every entry here records the key count its config produces, and `verify()` checks it. MorphoMNIST
is deliberately absent; those models are being retrained.
"""
from dataclasses import dataclass, field

BUCKET = "najibi-research-7f2a"


@dataclass(frozen=True)
class Model:
    name: str
    dataset: str          # adapter to use: "c3di" | "celebahq"
    config: str           # repo-relative config that reproduces the architecture
    s3_ckpt: str          # key of the final checkpoint, without the bucket
    n_keys: int           # model.state_dict() size, verified at build time
    graph: str = ""       # which causal graph, where a dataset has more than one
    attrs: tuple = ()     # conditioned attributes, in config order
    note: str = ""

    @property
    def uri(self) -> str:
        return f"s3://{BUCKET}/{self.s3_ckpt}"


_C = "experiments/hdae/configs"
_A = ("Male", "Young", "Beard", "Bald")
_B = ("Eyeglasses", "Smiling")
_F = ("class", "pos_spl", "pos_obj", "rot_obj")

MODELS = {m.name: m for m in [
    # --- Causal3DIdent: 4 conditioned factors, 3 further factors measured but never conditioned
    Model("c3di_k1", "c3di", f"{_C}/c3di_hier_k1_final.yaml",
          "able/causal3dident/v2_validated/hdae/c3di_k1_final-step217936.ckpt", 720, attrs=_F),
    Model("c3di_k5", "c3di", f"{_C}/c3di_hier_k5_final.yaml",
          "able/causal3dident/v2_validated/hdae/k5/c3di_k5_final-step192936.ckpt", 736, attrs=_F),
    Model("c3di_k11", "c3di", f"{_C}/c3di_hier_k11_final.yaml",
          "able/causal3dident/v2_validated/hdae/c3di_k11_final-step192936.ckpt", 760, attrs=_F),

    # --- CelebA-HQ graph A: Male/Young -> Beard/Bald
    Model("celebahq_a_k1", "celebahq", f"{_C}/celebahq256_k1_p2.yaml",
          "able/celebahq/graph_a/k1_p2/epoch=124-step=46875.ckpt", 827, graph="A", attrs=_A),
    Model("celebahq_a_k5", "celebahq", f"{_C}/celebahq256_k5_p2.yaml",
          "able/celebahq/graph_a/k5_p2/epoch=124-step=46875.ckpt", 843, graph="A", attrs=_A),
    Model("celebahq_a_k11", "celebahq", f"{_C}/celebahq256_k11_cd015r.yaml",
          "able/celebahq/graph_a/k11_cd015r/epoch=124-step=46875.ckpt", 867, graph="A", attrs=_A,
          note="the evaluated one; k11_cd015 is the run that went non-finite"),

    # --- CelebA-HQ graph B: two independent attributes, no edges
    Model("celebahq_b_k1", "celebahq", f"{_C}/celebahq256_b_k1_p2.yaml",
          "able/celebahq/graph_b/k1_p2/epoch=124-step=46875.ckpt", 823, graph="B", attrs=_B),
    Model("celebahq_b_k11", "celebahq", f"{_C}/celebahq256_b_k11_p2.yaml",
          "able/celebahq/graph_b/k11_p2/epoch=124-step=46875.ckpt", 863, graph="B", attrs=_B),
]}


def get(name: str) -> Model:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(MODELS))}")
    return MODELS[name]


def by_dataset(dataset: str):
    return [m for m in MODELS.values() if m.dataset == dataset]


def build(model: Model):
    """Construct the architecture its config describes. No weights loaded."""
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule
    cfg = load_hdae_config(os.path.join(root, model.config), require_data=False)
    return HDAELitModule(cfg.train_conf), cfg


def verify(model: Model) -> dict:
    """Does the config still produce the architecture this entry claims?

    Catches a config edited after the model was trained, which would otherwise surface as a
    silently half-initialised model at load time.
    """
    lit, cfg = build(model)
    n = len(lit.model.state_dict())
    enc = cfg.raw["encoder"]
    return {
        "name": model.name,
        "n_keys": n,
        "expected": model.n_keys,
        "ok": n == model.n_keys,
        "K": len(enc["hier_tap_block_ids"]),
        "n_attributes": enc["n_attributes"],
        "attr_embed_dim": enc["attr_embed_dim"],
        "dataset": cfg.raw["data"]["type"],
    }
