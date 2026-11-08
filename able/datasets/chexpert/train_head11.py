"""Measurement run: the same recipe as best.ckpt, with an 11th binary head.

enlarged_cardiomediastinum could not be gated in v2 because best.ckpt has no
head for it. This trains the identical architecture and recipe with that head
added, so the label can be gated on THIS project's classifier rather than on
XRV, which is not the scoring instrument.

Nothing here touches the frozen protocol:

  * best.ckpt is NOT overwritten -- this writes to its own --out directory.
  * chexpert_meta.csv is NOT modified. The 11th label column is built into a
    scratch copy of the meta from the checksummed source_csv/train.csv under
    the same eval-only convention build_meta.py step 4 uses (-1 -> 0,
    blank -> 0), and images_224.npy is reached by symlink.
  * the split, the fit/select/heldout partition and the seed are unchanged, so
    the 11-head trunk sees exactly the images the 10-head trunk saw.

The v0 held-out gate is disabled for this run (MODELED_BINARY = []). v2 gates
on radiologist_val, not on held-out report labels, and a held-out gate failure
here would halt a run whose only purpose is to produce a radiologist_val
number. The eleven held-out AUCs are still reported.

The other ten heads will shift slightly from best.ckpt: the trunk is shared and
now carries an eleventh loss term. That is what "same recipe plus a head"
means, and it is why this checkpoint is a measurement instrument for ONE label
and not a replacement for best.ckpt.
"""
import os, sys

# python -I drops the script directory from sys.path; put it back so the
# sibling module resolves however this is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_predictors as T

BINARY11 = T.BINARY + ["enlarged_cardiomediastinum"]

if __name__ == "__main__":
    T.BINARY = BINARY11
    T.MODELED_BINARY = []        # see the docstring: v2 does not gate on held-out
    print(f"heads: {len(BINARY11)} binary + 3-way age")
    print(f"       {BINARY11}")
    T.main()
