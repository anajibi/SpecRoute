export PYTHONPATH=$PWD
export NIMG=25
export T_EVAL=100

export ATTR_CLF=experiments/hdae/outputs/shared_attr_classifier.pt

export CFG_FLAT=experiments/hdae/configs/celeba64_flat.yaml
export CFG_K3=experiments/hdae/configs/celeba64_hier_k3.yaml
export CFG_K5=experiments/hdae/configs/celeba64_hier_k5.yaml
export CFG_K5_REV=experiments/hdae/configs/celeba64_hier_k5_reverse.yaml
export CFG_K5_EQ=experiments/hdae/configs/celeba64_hier_k5_equal.yaml

export CKPT_FLAT=experiments/hdae/outputs/celeba64_flat/checkpoints/last.ckpt
export CKPT_K3=experiments/hdae/outputs/celeba64_hier_k3/checkpoints/last.ckpt
export CKPT_K5=experiments/hdae/outputs/celeba64_hier_k5/checkpoints/last.ckpt
export CKPT_K5_REV=experiments/hdae/outputs/celeba64_hier_k5_reverse/checkpoints/last.ckpt
export CKPT_K5_EQ=experiments/hdae/outputs/celeba64_hier_k5_equal/checkpoints/last.ckpt

python experiments/hdae/run_cf_consistency.py \
  --cohorts experiments/hdae/outputs/cf_consistency/cohorts.json \
  --models \
    k3=experiments/hdae/configs/celeba64_hier_k3.yaml,experiments/hdae/outputs/celeba64_hier_k3/checkpoints/last.ckpt,experiments/hdae/outputs/celeba64_hier_k3/latent_probing/probes/probe_metrics.csv,experiments/hdae/outputs/celeba64_hier_k3/latent_probing/probes/weights \
    k5=experiments/hdae/configs/celeba64_hier_k5.yaml,experiments/hdae/outputs/celeba64_hier_k5/checkpoints/last.ckpt,experiments/hdae/outputs/celeba64_hier_k5/latent_probing/probes/probe_metrics.csv,experiments/hdae/outputs/celeba64_hier_k5/latent_probing/probes/weights \
    k5_equal=experiments/hdae/configs/celeba64_hier_k5_equal.yaml,experiments/hdae/outputs/celeba64_hier_k5_equal/checkpoints/last.ckpt,experiments/hdae/outputs/celeba64_hier_k5_equal/latent_probing/probes/probe_metrics.csv,experiments/hdae/outputs/celeba64_hier_k5_equal/latent_probing/probes/weights \
    k5_reverse=experiments/hdae/configs/celeba64_hier_k5_reverse.yaml,experiments/hdae/outputs/celeba64_hier_k5_reverse/checkpoints/last.ckpt,experiments/hdae/outputs/celeba64_hier_k5_reverse/latent_probing/probes/probe_metrics.csv,experiments/hdae/outputs/celeba64_hier_k5_reverse/latent_probing/probes/weights \
  --attr-classifier experiments/hdae/outputs/finetuned_attr_classifier.pt \
  --attributes Smiling,Eyeglasses,Male,Young \
  --directions positive,negative \
  --strength 4.0 \
  --T-eval 100 \
  --batch-size 32 \
  --cache-dir experiments/hdae/outputs/cf_consistency/cache \
  --out experiments/hdae/outputs/cf_consistency/cf_consistency_strength1.csv



#python experiments/hdae/counterfactuals/run_preservation_sweep.py \
#  --config "$CFG_K5" \
#  --ckpt "$CKPT_K5" \
#  --probe-metrics experiments/hdae/outputs/celeba64_hier_k5/latent_probing/probes_linear/probe_metrics.csv \
#  --probe-weights-dir experiments/hdae/outputs/celeba64_hier_k5/latent_probing/probes_linear/weights \
#  --attr-classifier "$ATTR_CLF" \
#  --attributes Smiling,Eyeglasses,Male,Young \
#  --strengths 0,0.5,1,2,4 \
#  --direction both \
#  --num-images "$NIMG" \
#  --batch-size 128 \
#  --T "$T_EVAL" \
#  --normalize-strength \
#  --per-attribute-matrix \
#  --save-grids \
#  --output-dir experiments/hdae/outputs/celeba64_hier_k5/counterfactuals/preservation_sweep_small
#
#python experiments/hdae/counterfactuals/run_preservation_sweep.py \
#  --config "$CFG_K5_REV" \
#  --ckpt "$CKPT_K5_REV" \
#  --probe-metrics experiments/hdae/outputs/celeba64_hier_k5_reverse/latent_probing/probes_linear/probe_metrics.csv \
#  --probe-weights-dir experiments/hdae/outputs/celeba64_hier_k5_reverse/latent_probing/probes_linear/weights \
#  --attr-classifier "$ATTR_CLF" \
#  --attributes Smiling,Eyeglasses,Male,Young \
#  --strengths 0,0.5,1,2,4 \
#  --direction both \
#  --num-images "$NIMG" \
#  --batch-size 128 \
#  --T "$T_EVAL" \
#  --normalize-strength \
#  --per-attribute-matrix \
#  --save-grids \
#  --output-dir experiments/hdae/outputs/celeba64_hier_k5_reverse/counterfactuals/preservation_sweep_small
#
#
#python experiments/hdae/counterfactuals/run_preservation_sweep.py \
#  --config "$CFG_K5_EQ" \
#  --ckpt "$CKPT_K5_EQ" \
#  --probe-metrics experiments/hdae/outputs/celeba64_hier_k5_equal/latent_probing/probes_linear/probe_metrics.csv \
#  --probe-weights-dir experiments/hdae/outputs/celeba64_hier_k5_equal/latent_probing/probes_linear/weights \
#  --attr-classifier "$ATTR_CLF" \
#  --attributes Smiling,Eyeglasses,Male,Young \
#  --strengths 0,0.5,1,2,4 \
#  --direction both \
#  --num-images "$NIMG" \
#  --batch-size 128 \
#  --T "$T_EVAL" \
#  --normalize-strength \
#  --per-attribute-matrix \
#  --save-grids \
#  --output-dir experiments/hdae/outputs/celeba64_hier_k5_equal/counterfactuals/preservation_sweep_small