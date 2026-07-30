#!/bin/bash

#echo "Launching k1 run..."
#nohup python experiments/hdae/scripts/run_full_pipeline.py \
#  --config /home/anajibi/HDM/experiments/hdae/configs/hier_k1.yaml \
#  --pcf-guidance-scale 8.0 > run_k1.log 2>&1 &

echo "Launching k5 run..."
nohup python experiments/hdae/scripts/run_full_pipeline.py \
  --config /home/anajibi/HDM/experiments/hdae/configs/hier_k5.yaml \
  --pcf-guidance-scale 8.0 > run_k5.log 2>&1 &

echo "Launching k11 run..."
nohup python experiments/hdae/scripts/run_full_pipeline.py \
  --config /home/anajibi/HDM/experiments/hdae/configs/hier_k11.yaml \
  --pcf-guidance-scale 8.0 > run_k11.log 2>&1 &

echo "All three processes launched in the background."
echo "Waiting for all to complete. You can tail the log files to check progress..."

# 'wait' ensures the script doesn't exit until all background jobs (&) are done
wait

echo "All parallel runs completed."
