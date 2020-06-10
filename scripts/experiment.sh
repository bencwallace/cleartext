#!/bin/bash
for tokens in 10000 30000
  for units in 50 250 500
    python -m train -e 40 -n 100000 -d 300 -t $tokens -r $units -a $units
  done
done
