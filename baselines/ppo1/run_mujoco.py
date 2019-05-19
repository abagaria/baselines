# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
import pdb
import pickle

from baselines import logger
import sys
from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
from simple_rl.tasks.fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
from simple_rl.tasks.ant_maze.AntMazeMDPClass import AntMazeMDP


def train(env_id, num_episodes, seed, num_options,app, saves ,wsaves, epoch,dc):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    mdp = AntMazeMDP(seed=seed, vary_init=True, dense_reward=True, render=False)
    env = mdp.env

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    if num_options ==1:
        optimsize=64
    elif num_options ==2:
        optimsize=32
    else:
        print("Only two options or primitive actions is currently supported.")
        sys.exit()

    overall_reward, overall_durations, validation_rewards, validation_durations = pposgd_simple.learn(env, policy_fn, 
            max_episodes=num_episodes,
            timesteps_per_batch=args.steps,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc
        )
    env.close()

    with open("{}_training_scores_{}.pkl".format(args.env, seed), "wb+") as f:
        pickle.dump(overall_reward, f)
    with open("{}_training_durations_{}.pkl".format(args.env, seed), "wb") as f:
        pickle.dump(overall_durations, f)
    with open("{}_validation_scores_{}.pkl".format(args.env, seed), "wb+") as f:
        pickle.dump(validation_rewards, f)
    with open("{}_validation_durations_{}.pkl".format(args.env, seed), "wb+") as f:
        pickle.dump(validation_durations, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--epoch', help='Epoch', type=int, default=-1) 
    parser.add_argument('--dc', type=float, default=0.)
    
    # Specification for episodic MDP tasks:
    # Number of episodes in the MDP, and the number of steps per episodes
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)

    args = parser.parse_args()

    train(args.env, num_episodes=args.episodes, seed=args.seed, num_options=args.opt, app=args.app, saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,dc=args.dc)
