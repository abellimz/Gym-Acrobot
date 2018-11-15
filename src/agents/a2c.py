import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.agents.agent_v2 import AgentV2


class A2CAgent(AgentV2):
  def __init__(self, action_size, state_size,
               gamma=.99, nb_steps_warmup=10, actor_lr=0.001, critic_lr=0.002,
               train_interval=1):
    super().__init__()

    self.value_size = 1
    self.action_size = action_size
    self.state_size = state_size
    self.gamma = gamma
    self.nb_steps_warmup = nb_steps_warmup
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr
    self.actor = self.build_actor()
    self.critic = self.build_critic()
    self.compiled = True
    self.train_interval = train_interval
    self.last_action = None
    self.step = 0

  # using the output of policy network, pick action stochastically
  def select_action(self, state):
    policy = self.actor.predict(state).flatten()
    if self.training:
      return np.random.choice(len(policy), 1, p=policy)[0]
    else:
      return np.argmax(policy)

  def forward(self, observation):
    """Takes the an observation from the environment and returns the action to be taken next.
    If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

    # Argument
        observation (object): The current observation from the environment.

    # Returns
        The next action to be executed in the environment.
    """
    self.state = np.expand_dims(observation, 0)
    action = self.select_action(self.state)
    self.last_action = action
    return action

  def backward_obs(self, reward, observation, terminal=False):
    """Updates the agent after having executed the action returned by `forward`.
    If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
    """
    if self.step <= self.nb_steps_warmup or self.step % self.train_interval != 0:
      return

    target = np.zeros((1, 1))
    advantages = np.zeros((1, self.action_size))

    value = self.critic.predict(self.state)[0]
    next_value = self.critic.predict(np.expand_dims(observation, 0))[0]

    if terminal:
      advantages[0][self.last_action] = reward - value
      target[0][0] = reward
    else:
      advantages[0][self.last_action] = reward + self.gamma * (next_value) - value
      target[0][0] = reward + self.gamma * next_value

    self.actor.fit(self.state, advantages, epochs=1, verbose=0)
    self.critic.fit(self.state, target, epochs=1, verbose=0)

  def compile(self, optimizer, metrics=[]):
    """Compiles an agent and the underlaying models to be used for training and testing.

    # Arguments
        optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
        metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
    """
    pass

  # approximate policy and value using Neural Network
  # actor: state is input and probability of each action is output of model
  def build_actor(self):
    actor = Sequential()
    actor.add(Dense(24, input_dim=self.state_size, activation='relu'))
    actor.add(Dense(self.action_size, activation='softmax'))
    actor.summary()
    # See note regarding crossentropy in cartpole_reinforce.py
    actor.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=self.actor_lr))
    return actor

  # critic: state is input and value of state is output of model
  def build_critic(self):
    critic = Sequential()
    critic.add(Dense(24, input_dim=self.state_size, activation='relu'))
    critic.add(Dense(self.value_size, activation='linear'))
    critic.summary()
    critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
    return critic
