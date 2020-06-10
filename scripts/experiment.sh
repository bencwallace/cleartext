#!/bin/bash
for units in 50 100 250 300 500
do
  python -m train -e 40 -n 100000 -d 50 -t 100 -r $units -a $units
done
