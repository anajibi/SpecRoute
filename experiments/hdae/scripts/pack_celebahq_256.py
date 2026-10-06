"""Pack CelebA-HQ to a 256x256 raw-uint8 LMDB.

256 is the ceiling: it is the largest DiffAE template that exists (ffhq256_autoenc) and the
largest pretrained checkpoint on offer. Source images are 1024x1024, so this discards detail,
but no 512/1024 DiffAE model exists to fine-tune from.

lanczos rather than the default bicubic: this is a 4x downscale, where lanczos is the better
antialiasing kernel. preprocess() is resumable via the __completed__ key, so an interruption
costs only the in-flight batch.
"""
import sys, time
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.data.preprocess import preprocess

D = "/home/exouser/SpecRoute/experiments/hdae/data"
t0 = time.time()
meta = preprocess(
    image_dir=f"{D}/celebahq_src/images",
    lmdb_path=f"{D}/packed/celebahq_256.lmdb",
    attr_path=f"{D}/celebahq_src/list_attr_celeba.txt",
    partition_path=f"{D}/celebahq_src/list_eval_partition.txt",
    attr_npz=f"{D}/packed/celebahq_256_attrs.npz",
    image_size=256,
    resize_filter="lanczos",
    num_workers=8,
)
print(f"packed {meta['num_images']} images at {meta['image_size']}px in {(time.time()-t0)/60:.1f} min")
print(f"alignment case: {meta['alignment_case']}  (A = ids matched directly, no mapping needed)")
print(f"attributes: {len(meta['attribute_names'])}")
