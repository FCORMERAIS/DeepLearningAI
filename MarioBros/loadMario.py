#Program for load AI for see it play super_mario_bros

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env,SIMPLE_MOVEMENT)
env = GrayScaleObservation(env,keep_dim=True)
env = DummyVecEnv([lambda : env])
env = VecFrameStack(env, 4,channels_order='last')

model = PPO.load('./train/best_model2_500000')
state = env.reset()
while True :
    action,_ = model.predict(state)
    state,reward,done,info = env.step(action)
    env.render()
