import json
from pathlib import Path
import numpy as np, pytest
pytest.importorskip('lmdb'); Image=pytest.importorskip('PIL.Image')
from experiments.hdae.data.preprocess import preprocess
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.data.datamodule import CelebAHQDataModule

def sources(root):
    images=root/'images';images.mkdir();names=[f'A{i}' for i in range(40)]
    with open(root/'attr.txt','w') as a, open(root/'part.txt','w') as p:
        a.write('10\n'+' '.join(names)+'\n')
        for i in range(10):
            Image.new('RGB',(12,12),(i,i,i)).save(images/f'{i}.png')
            vals=[1 if (i+j)%2 else -1 for j in range(40)];a.write(f'{i}.jpg '+' '.join(map(str,vals))+'\n');p.write(f'{i}.jpg {i%3}\n')
    return images,root/'attr.txt',root/'part.txt'

def test_preprocess_dataset_alignment_and_lazy_env(tmp_path):
    images,attr,part=sources(tmp_path);lmdb=tmp_path/'packed.lmdb';npz=tmp_path/'attrs.npz'
    preprocess(images,lmdb,attr,part,npz,8,'bicubic',1);ds=CelebAHQPacked(lmdb,npz)
    assert ds._env is None;item=ds[3];assert item['img'].shape==(3,8,8) and -1<=item['img'].min()<=item['img'].max()<=1
    expected=np.asarray([1 if (3+j)%2 else -1 for j in range(40)],dtype=np.int8);assert np.array_equal(item['attr'].numpy(),expected)
    dm=CelebAHQDataModule(lmdb,npz,2,0);dm.setup();assert all(len(x)>0 for x in [dm.train_set,dm.val_set,dm.test_set])
