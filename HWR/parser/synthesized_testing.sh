#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python3 predict.py 45 > parser/results_original

#| tee parser/results_original

for i in {1..20}
do
   sed -i '1,100d' parser/train_OCR_560.part
   python3 predict.py 45 > parser/results_$i
done