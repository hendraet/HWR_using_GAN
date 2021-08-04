#!/bin/bash

i=150
for j in {1..10}
do
  for x in {1..5}
  do
    python3 synthesized_training_input_imgs.py --model 3050 --n_generated_images $i | tee -a results_3050_${i};
    python3 synthesized_training_input_imgs.py --model 4050 --n_generated_images $i | tee -a  results_4050_${i};
  done
  let i=i+20
done
