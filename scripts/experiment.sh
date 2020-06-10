#!/bin/bash
for tokens in 10000 30000; do
  for units in 50 250 500; do
    # todo: write this in python
    # todo: print which run this is
    python -m train -e 40 -n 100000 -d 300 -t $tokens -r $units -a $units
  done
done
