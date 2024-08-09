
start_exp=-2.0 # -2!!!
end_exp=0.0
step=0.1
n_steps=20
results_dir=results
models_file=configs/all_configs.json

for sup_size in $(seq -2.0 0.1 0.0); do
    sup=$(python -c "print(10.0 ** $sup_size)")
    python run.py --sg-size=$sup --models-file=$models_file --perturbation=0.25 --n-runs=100 --outdir=$results_dirs &
done
