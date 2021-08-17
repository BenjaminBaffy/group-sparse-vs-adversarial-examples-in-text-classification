#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
DATASET="sst2"
RECIPE="textfooler"
MODEL="roberta-base-${DATASET}"
for recipe in $RECIPE
do
  for model in $MODEL
  do
    LOG_FILE_NAME="${model}_${recipe}"
    textattack attack --model $model --num-examples 1821 --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --model-batch-size 128 --recipe $recipe \
     --num-workers-per-device 16 --dataset-from-file sst2_dataset.py\
     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"
  done
done


#RECIPE="tf-adj"
#for recipe in $RECIPE
#do
#  for model in $MODEL
#  do
#     LOG_FILE_NAME="${model}_${recipe}"
#    textattack attack --model $model --num-examples 1821 --attack-from-file recipes/textfooler_jin_2019_adjusted.py \
#    --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --dataset-from-file sst2_dataset.py --model-batch-size 64\
#     --num-workers-per-device 16\
#     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"
#  done
#done
