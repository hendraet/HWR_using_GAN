#!/bin/bash


for j in {1..3}
do
  for i in {1..10000..200}
  do
    python3 create_synthesized_images.py 560 $i;
    python3 create_synthesized_images.py 520 $i;
    python3 create_synthesized_images.py 537 $i;
    python3 create_synthesized_images.py 599 $i;
  done
done
