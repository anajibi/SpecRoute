import csv,random
from pathlib import Path
import torch,yaml
def load_config(path):
    with open(path) as f:return yaml.safe_load(f)
def seed_all(seed): random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def ensure_output(cfg):
    p=Path(__file__).parents[1]/cfg['train']['output_dir']; p.mkdir(parents=True,exist_ok=True); return p
def save_csv(rows,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with open(path,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def levels(out,k): return [out[f'Z{i}'] for i in range(k,0,-1)]
def build_trainable(k):
    from .evidence import EvidencePyramid
    from .encoders import HierarchicalEncoderK3,HierarchicalEncoderK5
    from .decoders import DeterministicLatentDecoder,LatentDecoderDiffusion32x32
    return EvidencePyramid(),(HierarchicalEncoderK3() if k==3 else HierarchicalEncoderK5()),DeterministicLatentDecoder(k),LatentDecoderDiffusion32x32(k)
