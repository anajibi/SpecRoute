"""CelebA-HQ-compatible image and extracted-latent datasets."""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self,root,image_size=256,max_images=None):
        self.root=Path(root); self.files=sorted(p for p in self.root.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
        if max_images:self.files=self.files[:max_images]
        if not self.files: raise FileNotFoundError(f"No images found under {root}")
        from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
        self.transform=Compose([Resize(image_size),CenterCrop(image_size),ToTensor(),Normalize([.5]*3,[.5]*3)])
    def __len__(self):return len(self.files)
    def __getitem__(self,i):return self.transform(Image.open(self.files[i]).convert("RGB")),str(self.files[i])

class SyntheticImageDataset(Dataset):
    def __init__(self,n=100,image_size=256,seed=0):self.n,self.image_size,self.seed=n,image_size,seed
    def __len__(self):return self.n
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(self.seed+i);return torch.rand(3,self.image_size,self.image_size,generator=g)*2-1,f"synthetic_{i:06d}"

class LatentDataset(Dataset):
    def __init__(self,path):self.data=torch.load(path,map_location="cpu",weights_only=False);self.zs=self.data["latents"]
    def __len__(self):return self.zs[0].shape[0]
    def __getitem__(self,i):return tuple(z[i] for z in self.zs)

def image_loader(cfg,split="train",max_images=None,shuffle=None):
    d=cfg["dataset"]; root=d.get(f"{split}_root",d.get("root")); ds=SyntheticImageDataset(max_images or d.get("synthetic_size",100),d["image_size"],cfg["seed"]) if d.get("synthetic",False) else ImageDataset(root,d["image_size"],max_images)
    batch=cfg["stage1"].get("batch_size",32);return DataLoader(ds,batch_size=batch,shuffle=(split=="train" if shuffle is None else shuffle),num_workers=d.get("num_workers",4),pin_memory=True)
