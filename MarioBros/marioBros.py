#Program for train AI for play to super_mario_bros

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


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model2_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env,SIMPLE_MOVEMENT)
env = GrayScaleObservation(env,keep_dim=True)
env = DummyVecEnv([lambda : env])
env = VecFrameStack(env, 4,channels_order='last')

state = env.reset()
#for i in range(4) :
#    state,reward,done,info = env.step([env.action_space.sample()])
#    plt.figure(figsize=(10,8))
#    for idx in range(state.shape(3)):
#        plt.subplot(1, 4,idx+1)
#        plt.imshow(state[0][:,:,idx])
#    plt.show()

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
model = PPO.load('./train/best_model2_120000')

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

model = PPO('CnnPolicy',env,verbose=1,tensorboard_log=LOG_DIR,learning_rate=0.000002,n_steps=512)
model.learn(total_timesteps=500000,callback=callback)

model.save("ModelMarioAI")