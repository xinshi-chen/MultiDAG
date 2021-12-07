#!/bin/bash

p=100  # Nodes
e=10  # Edge perturbations per task
nh=0.05  # SERGIO hill coefficient
K_vals=(1 1 10)  # Tasks
n_task_vals=(1000 100 100)  # Samples per task, must be the same length as K
rho_vals=(0.1)  # Group-norm weight in recovery objective (recovery hyperparam)
alpha=0.001
threshold_vals=(0.8)  # Threshold for true vs. spurious edges (comparison hyperparam)

for i in ${!K_vals[*]}; do
  K=${K_vals[i]}
  n_task=${n_task_vals[i]}
  n=$((K*n_task))  # Total samples
  echo "######################################################################"
  echo "Running SERGIO for K-${K} p-${p} e-${e} n-${n} nh-${nh}"
  echo "######################################################################"
  cd ../sergio_scripts
  python generator.py -K ${K} -e ${e} -n ${n_task} -nh ${nh}

  cd ../experiments
  sim_file="sergio_K-${K}_p-${p}_e-${e}_n-${n}_nh-${nh}.npz"
  real_dir="../sergio_scripts/sergio_output/${sim_file}"

  for rho in "${rho_vals[@]}"; do
    for threshold in "${threshold_vals[@]}"; do
      echo "######################################################################"
      echo "Recovering SERGIO for K-${K} n-${n} rho-${rho} thresh-${threshold}"
      echo "######################################################################"
      python main_linear.py --real_dir ${real_dir} --group_size ${K} \
      -rho ${rho} -K ${K} -p ${p} -n_sample ${n} -num_epochs 3000 \
      -alpha ${alpha} -gpu 0

      python sergio_comparison.py --real_dir ${real_dir} -rho ${rho} \
      -K ${K} -alpha ${alpha} --threshold ${threshold}
    done
  done
done
