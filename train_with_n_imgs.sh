#!/bin/bash


for j in {1..3}
do
  for i in {5001..9001..1000}
  do
    python3 synthesized_training_IAM.py 560 $i;
    python3 synthesized_training_IAM.py 520 $i;
    python3 synthesized_training_IAM.py 537 $i;
    python3 synthesized_training_IAM.py 599 $i;
  done
done
