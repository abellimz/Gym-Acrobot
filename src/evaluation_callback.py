from collections import namedtuple

from rl.callbacks import Callback

Result = namedtuple('Result', ['best', 'worst', 'average'])

class EvaluationCallback(Callback):
  def __init__(self, verbose):
    # Some algorithms compute multiple episodes at once since they are multi-threaded.
    # We therefore use a dictionary that is indexed by the episode to separate episodes
    # from each other.
    super().__init__()
    self.verbose = verbose
    self.observations = {}
    self.episodes_steps = {}
    self.step = 0
    self.evaluation_result = None

  def get_result(self):
    return self.evaluation_result

  def on_train_begin(self, logs):
    """ Print training values at beginning of training """

  def on_train_end(self, logs):
    """ Print evaluation information at end of training """
    shortest_steps = 10000000
    longest_steps = 0
    total_steps = 0
    for episode, steps in self.episodes_steps.items():
      shortest_steps = min(steps, shortest_steps)
      longest_steps = max(steps, longest_steps)
      total_steps += steps
    average_steps = total_steps / len(self.episodes_steps)
    self.evaluation_result = Result(shortest_steps, longest_steps, average_steps)
    if self.verbose:
      print('| Shortest Steps | Longest Steps | Average Steps |')
      print('| {} | {} | {} |'.format(shortest_steps, longest_steps, average_steps))

  def on_episode_begin(self, episode, logs):
    """ Reset environment variables at beginning of each episode """
    self.observations[episode] = []

  def on_episode_end(self, episode, logs):
    """ Compute and print training statistics of the episode when done """
    self.episodes_steps[episode] = len(self.observations[episode])
    del self.observations[episode]

  def on_step_end(self, step, logs):
    """ Update statistics of episode after each step """
    episode = logs['episode']
    self.observations[episode].append(logs['observation'])
    self.step += 1
