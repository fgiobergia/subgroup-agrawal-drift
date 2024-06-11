
start_exp=-2.0 # -2!!!
end_exp=0.0
step=0.1
n_steps=20

for sup_size in $(seq -2.0 0.1 0.0); do
    sup=$(python -c "print(10.0 ** $sup_size)")
    python run.py --sg-size=$sup --models-file=best_config.json --perturbation=0.25
done
