#!/bin/bash
for seed in {0..4}
do
        echo $seed
        python run_mujoco.py --env="reacher" --saves --wsaves --opt 2 --seed $seed --app savename --dc 0.1 --episodes=5000 --steps=2000
done