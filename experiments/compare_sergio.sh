#!/bin/bash

K=1
p=100
e=10
n=100
nh=2.0
rho=(1)
threshold=0.04

sim_file="sergio_K-${K}_p-${p}_e-${e}_n-${n}_nh-${nh}.npz"

python comparison.py \
  --real_dir ${sim_file} \
  -rho ${rho[i]} \
  -K ${K} \
  --threshold ${threshold}