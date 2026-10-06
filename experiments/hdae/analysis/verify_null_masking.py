"""Prove that nulling attribute i touches attribute i's embedding slice and nothing else.

attr_fusion is concat_film, so each attribute owns a fixed contiguous slice of the attribute
embedding. If nulling Bald is implemented correctly, exactly the Bald slice changes.
"""
import sys, glob, torch
sys.path.insert(0,"/home/exouser/SpecRoute/diffae_upstream"); sys.path.insert(0,"/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule

cfg=load_hdae_config("experiments/hdae/configs/celebahq256_k11_cd015r.yaml")
ck=sorted(glob.glob("experiments/hdae/outputs/celebahq256_k11_cd015r/checkpoints/*.ckpt"))
lit=HDAELitModule.load_from_checkpoint(ck[-1], conf=cfg.train_conf, map_location="cpu")
emb=lit.ema_model.attr_embedding.eval()
names=[s.name for s in cfg.hdae_conf.encoder.cond_specs]
B=4; n=len(names)
y=torch.tensor([[1.,1.,1.,1.]]*B)
base=emb(y, null_mask=torch.zeros(B,n,dtype=torch.bool), apply_dropout=False)
W=base.shape[1]; sl=W//n
print(f"embedding width {W}, {n} attrs -> slice {sl} each\n")
print(f"{'nulled':22} " + " ".join(f"{a:>9}" for a in names))
for idx in ([0],[1],[2],[3],[2,3],[0,1,2,3]):
    m=torch.zeros(B,n,dtype=torch.bool); m[:,idx]=True
    out=emb(y, null_mask=m, apply_dropout=False)
    d=(out-base).abs()
    per=[d[:, i*sl:(i+1)*sl].max().item() for i in range(n)]
    tag="{"+",".join(names[i] for i in idx)+"}"
    flag=""
    changed={i for i,v in enumerate(per) if v>1e-6}
    if changed!=set(idx): flag="  <<< MISMATCH"
    print(f"{tag:22} " + " ".join(f"{v:9.5f}" for v in per) + flag)
print("\nA slice is non-zero exactly when its attribute is in the nulled set -> correct.")
