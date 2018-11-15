from __future__ import print_function

import argparse
import sys

import gym
from keras import Input, Model

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Concatenate
from keras.optimizers import Adam

from rl.agents import SARSAAgent, DQNAgent, DDPGAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from scipy.stats import bayes_mvs

from src.agents.a2c import A2CAgent
from src.agents.a2c_lstm import A2CAgent_LSTM
from src.evaluation_callback import EvaluationCallback

ENV_NAME = 'Acrobot-v1'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

# Print information about environment
print("#### ENVIRONMENT ####")
print("Observation space:", env.observation_space, "High:", env.observation_space.high, "low",
      env.observation_space.low)
print("Action space:", env.action_space)
print("#####################")


def get_model(model_type):
  model = None
  if model_type == "mlp":
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
  else:
    print("Unsupported model")
    exit(1)
  print(model.summary())
  return model


def get_agent(agent_type, model_type, lr):
  if agent_type == "sarsa":
    policy = BoltzmannQPolicy()
    model = get_model(model_type)
    agent = SARSAAgent(model=model,
                       policy=policy,
                       nb_actions=nb_actions,
                       nb_steps_warmup=10,
                       gamma=0.99)
    agent.compile(Adam(lr), metrics=['mae'])
    return agent
  elif agent_type == "dqn":
    policy = BoltzmannQPolicy()
    model = get_model(model_type)
    memory = SequentialMemory(limit=50000, window_length=1)
    agent = DQNAgent(model=model,
                     policy=policy,
                     nb_actions=nb_actions,
                     memory=memory,
                     nb_steps_warmup=10,
                     target_model_update=1e-2,
                     enable_double_dqn=True)
    agent.compile(Adam(lr), metrics=['mae'])
    return agent
  elif agent_type == "a2c":
    agent = A2CAgent(nb_actions,
                     len(env.observation_space.high),
                     nb_steps_warmup=10,
                     actor_lr=0.001,
                     critic_lr=0.005)
    agent.compile(Adam(lr))
    return agent
  elif agent_type == "ppo":
    pass
  else:
    print("Unsupported model")
    exit(1)


def main(args):
  # Next, we build a very simple model.
  bests = []
  worsts = []
  averages = []
  for i in range(args.n):
    print("Training and evaluating iteration: ", i)
    # SARSA does not require a memory.
    agent = get_agent(args.agent, args.model, args.lr)
    agent.fit(env, nb_steps=args.train_steps, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    # sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    callback = EvaluationCallback(verbose=True)
    agent.test(env,
               nb_episodes=args.test_episodes,
               callbacks=[callback],
               verbose=0,
               visualize=False)
    best, worst, average = callback.get_result()
    bests.append(best)
    worsts.append(worst)
    averages.append(average)

  bests_mvs = bayes_mvs(bests)
  worsts_mvs = bayes_mvs(worsts)
  average_mvs = bayes_mvs(averages)
  print("Best Performance: {:.2f} +- {:.2f}".format(bests_mvs[0][0], bests_mvs[0][0] - bests_mvs[0][1][0]))
  print("Worst Performance: {:.2f} +- {:.2f}".format(worsts_mvs[0][0], worsts_mvs[0][0] - worsts_mvs[0][1][0]))
  print("Average Performance: {:.2f} +- {:.2f}".format(average_mvs[0][0], average_mvs[0][0] - average_mvs[0][1][0]))


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  # Data files args
  parser.add_argument('--n',
                      type=int,
                      default=20,
                      help='Number of times to repeat experiment')
  parser.add_argument('--train_steps',
                      type=int,
                      default=50000,
                      help='Number of training steps per experiment')
  parser.add_argument('--lr',
                      type=float,
                      default=0.001,
                      help='Learning rate to be used')
  parser.add_argument('--test_episodes',
                      type=int,
                      default=100,
                      help='Number of training steps per experiment')
  parser.add_argument('--agent',
                      type=str,
                      default="sarsa",
                      help='Agent to use for learning')
  parser.add_argument('--model',
                      type=str,
                      default="mlp",
                      help='model to use for Q approximation')
  return parser.parse_args(argv)


if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
