import numpy as np
import gym
from gym import spaces
import BS_association_ENV
import comparison_ENV
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, plot_results
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        num_UE = 10
        num_BS = 4
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(130,), dtype=np.float32)
        self.env = comparison_ENV.BsAssociation(BS_num=num_BS, UE_num=num_UE)
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return np.array(observation, dtype=np.float32), reward, done, info
    def reset(self):
        observation = self.env.reset()
        return np.array(observation, dtype=np.float32)
    def close (self):
        print("close")

# Main 함수
if __name__ == '__main__':
    env = CustomEnv()
    # env2 = Monitor(env)
 
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./DRL/logs/")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs")
    eval_callback = EvalCallback(env, eval_freq=100, deterministic=True, render=False,best_model_save_path="./logs/")
   
    model.learn(total_timesteps=10000000, callback=[eval_callback])
    model.save("ppo_v2")
    results = load_results("./logs/")
    plot_results(results, title="My Training Results")
    
    