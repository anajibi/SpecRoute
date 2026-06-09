"""Folder dataset returning image tensors in [-1,1]."""
from pathlib import Path
from torch.utils.data import DataLoader,Dataset
class ImageFolderDataset(Dataset):
    def __init__(self,root,image_size=256,max_images=None):
        from torchvision import transforms
        self.files=[p for p in Path(root).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}][:max_images]
        self.transform=transforms.Compose([transforms.Resize(image_size),transforms.CenterCrop(image_size),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        from PIL import Image
        return {'image':self.transform(Image.open(self.files[i]).convert('RGB')),'image_id':self.files[i].stem}
def make_loader(root,batch_size=4,image_size=256,max_images=None,shuffle=False,num_workers=0): return DataLoader(ImageFolderDataset(root,image_size,max_images),batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
