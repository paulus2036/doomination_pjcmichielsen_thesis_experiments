#!/bin/bash

# Define parameter values
methods=("PPO" "PPOPID" "PPOSaute" "PPOLag" "PPOCost" "P3O")
envs=("armament_burden" "volcanic_venture" "remedy_rush" "collateral_damage" "precipice_plunge" "detonators_dilemma")
#env_mods=("dark" "floor")
levels=(1 2 3)
seeds=(1 2 3 4 5)


# Loop over parameter combinations and submit Slurm jobs
for algo in "${methods[@]}"; do
    for env in "${envs[@]}"; do
        for level in "${levels[@]}"; do
            for seed in "${seeds[@]}"; do
                for env_mod in "${env_mods[@]}"; do
                    echo "Submitting job for combination: Algo=$algo, Env=$env, Level=$level, Seed=$seed"

                    # Create an SBATCH script
                    cat <<EOF | sbatch
#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 05:00:00
#SBATCH --gres gpu:1
#SBATCH -o /home/ttomilin/slurm/%j_"${algo}"_"${env}"_Level_${level}_$(date +%Y-%m-%d-%H-%M-%S).out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safety_doom
if [ \$? -eq 0 ]; then
    python3 ~/safety-doom/sample_factory/doom/train_vizdoom.py \
        --algo "$algo" \
        --env "$env" \
        --level $level \
        --seed $seed \
        --train_for_env_steps 500000000 \
        --heatmap_log_interval 10000000 \
        --with_wandb True \
        --wandb_tags TEST \
        --wandb_user tu-e \
        --wandb_project safety-doom
else
    echo "Failed to activate conda environment."
fi
conda deactivate
EOF
                done
            done
        done
    done
done

