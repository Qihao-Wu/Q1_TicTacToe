# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A state-settable environment for Tic-Tac-Toe game."""

import copy
import numpy as np
import random
from itertools import cycle
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep

########################################## Auxiliary Functions ##################################################
def print_tic_tac_toe(state):
    table_str = '''
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    - - + - - + - - + - - + - - + - - + - - + - - + - -
    {} | {} | {} | {} | {} | {} | {} | {} | {}
    '''.format(*tuple(state.flatten()))
    table_str = table_str.replace('0', ' ')
    table_str = table_str.replace('1', 'X')
    table_str = table_str.replace('2', 'O')
    print(table_str)

def print_traj(traj):
    steps = tf.concat(list(traj), axis=-1)
    table_str = '''Trajectory:
                       Start                                            End
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}  ->   {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    - + - + - + - + - + - + - + - + -       - + - + - + - + - + - + - + - + -
    {} | {} | {} | {} | {} | {} | {} | {} | {}       {} | {} | {} | {} | {} | {} | {} | {} | {}
    '''.format(*tuple(steps.numpy().flatten()))
    table_str = table_str.replace('0', ' ')
    table_str = table_str.replace('1', 'X')
    table_str = table_str.replace('2', 'O')
    print(table_str)

def ttt_action_fn(player, action):
    return {'position': action, 'value': player}

def p2_reward_fn(ts: TimeStep) -> float:
    return -ts.reward

def training_episode(tf_ttt_env, player_1, player_2):
    ts = tf_ttt_env.reset()
    player_1.reset()
    player_2.reset()
    time_steps = []
    if bool(random.randint(0, 1)):
        players = cycle([player_1, player_2])
    else:
        players = cycle([player_2, player_1])
    while not ts.is_last():
        player = next(players)
        player.act(collect=True)
        ts = tf_ttt_env.current_time_step()
        time_steps.append(ts)
    print('Episode Completed')
    return time_steps
#####################################################################################################################
class TicTacToeEnvironment(py_environment.PyEnvironment):
  """A state-settable environment for Tic-Tac-Toe game.

  For MCTS/AlphaZero, we need to keep states of the environment in a node and
  later restore them once MCTS selects which node to visit. This requires
  calling into get_state() and set_state() functions.

  The states are a 9 x 9 array (originally, 3x3) where 0 = empty, 1 = player, 2 = opponent.
  The action is a flattened scalar (originally, 2-d vector) to indicate the position for the player's move.
  """

  REWARD_WIN = np.asarray(10, dtype=np.float32)
  REWARD_LOSS = np.asarray(-10, dtype=np.float32)
  REWARD_DRAW_OR_NOT_FINAL = np.asarray(0.0, dtype=np.float32)
  # A very small number such that it does not affect the value calculation.
  # But due to the strong stochasticity, let's set to all 0.
  REWARD_ILLEGAL_MOVE = np.asarray(0, dtype=np.float32)  # can be -1
  REWARD_STOCHASTIC_MOVE = np.asarray(0, dtype=np.float32)  # can be -0.001

  REWARD_WIN.setflags(write=False)
  REWARD_LOSS.setflags(write=False)
  REWARD_DRAW_OR_NOT_FINAL.setflags(write=False)

  def __init__(self, rng: np.random.RandomState = None, discount=np.asarray(1.0, dtype=np.float32)):
      self._states = None
      self._discount = discount
      self.X_win = False
      self.O_win = False

  def action_spec(self):
      position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=890)
      # 81 x 11 to indicate 11 used energy levels
      value_spec = BoundedArraySpec((1,), np.int32, minimum=1, maximum=2)
      return {
          'position': position_spec,
          'value': value_spec,
      }

  def observation_spec(self):
      return {
        'board': BoundedArraySpec((9, 9), np.float32, minimum=0, maximum=2),
        'energy': BoundedArraySpec((2,), np.float32, minimum=0, maximum=10)
      }

  def _reset(self):
      self.X_win = False
      self.O_win = False
      self._states = {'board': np.zeros((9, 9), np.float32), 'energy': np.array([10, 10], dtype=np.float32)}
      return TimeStep(
          StepType.FIRST,
          np.asarray(0.0, dtype=np.float32),
          self._discount,
          self._states,
      )

  def _legal_actions(self, states: np.ndarray):
    return list(zip(*np.where(states == 0)))

  def _opponent_play(self, states: np.ndarray):
    actions = self._legal_actions(np.array(states))
    if not actions:
      raise RuntimeError('There is no empty space for opponent to play at.')

    if self._rng:
      i = self._rng.randint(len(actions))
    else:
      i = 0
    return actions[i]

  def get_state(self) -> TimeStep:
    # Returning an unmodifiable copy of the state.
    return copy.deepcopy(self._current_time_step)

  def set_state(self, time_step: TimeStep):
    self._current_time_step = time_step
    self._states = time_step.observation

  def _step(self, action: np.ndarray):
      if self._current_time_step.is_last():
          return self._reset()
      action_position = action['position'] // 11  # locate this time.
      action_energy = action['position'] // 81 / 10  # Positions x Energy levels. Energy used this time.
      # another way: (action % 11) / 10.0: One position x Energy levels, and to 81 positions.
      print('Energy used:', action_energy)
      if action_energy > 1:
          raise ValueError('Energy used cannot be greater than 1, but was {}'.format(action_energy))

      if action['value'] == 1:
          if self._states['energy'][0] <= action_energy:
              action_energy = self._states['energy'][0]
              print('No enough energy, use all the left:', action_energy)
          self._states['energy'][0] -= action_energy
      elif action['value'] == 2:
          if self._states['energy'][1] <= action_energy:
              action_energy = self._states['energy'][1]
              print('No enough energy, use all the left:', action_energy)
          self._states['energy'][1] -= action_energy
      else:
          raise ValueError('Error: action value is neither 1 nor 2')

      # The uncertainty of the action:
      if random.random() < 8/9 - 6/9 * action_energy:
          chosen_action = action_position
          action_position = chosen_action + random.choice([-10, -9, -8, -1, 1, 8, 9, 10])  # 1/8
          if action_position < 0 or action_position > 80 or abs(chosen_action // 9 - action_position // 9) > 1:
              print('Stochastic action chosen:', chosen_action, (chosen_action // 9, chosen_action % 9),
                    'Stochastic action executed:', action_position, (action_position // 9, action_position % 9),
                    'Illegal action: OUTSIDE!!!')
              return TimeStep(StepType.MID,
                              TicTacToeEnvironment.REWARD_STOCHASTIC_MOVE,
                              self._discount,
                              self._states)
          else:
              index_flat = np.array(range(81)) == action_position
              index = index_flat.reshape(self._states['board'].shape) == True
              if self._states['board'][index] != 0:
                  print('Stochastic action chosen:', chosen_action, (chosen_action // 9, chosen_action % 9),
                        'Stochastic action executed:', action_position, (action_position // 9, action_position % 9),
                        'Illegal action: OCCUPIED!!!')
                  if 0 in self._states['board']:
                      return TimeStep(StepType.MID,
                                      TicTacToeEnvironment.REWARD_STOCHASTIC_MOVE,
                                      self._discount,
                                      self._states)
                  else:
                      return TimeStep(StepType.LAST,
                                      TicTacToeEnvironment.REWARD_STOCHASTIC_MOVE,
                                      self._discount,
                                      self._states)
              else:
                  print('Stochastic action chosen:', chosen_action, (chosen_action // 9, chosen_action % 9),
                        'Stochastic action executed:', action_position, (action_position // 9, action_position % 9))
                  self._states['board'][index] = action['value']
                  is_final, reward = self._check_states(self._states['board'])
                  if np.all(self._states['board'] == 0):
                      step_type = StepType.FIRST
                  elif is_final:
                      step_type = StepType.LAST
                  else:
                      step_type = StepType.MID
                  return TimeStep(step_type, reward, self._discount, self._states)
      else:
          index_flat = np.array(range(81)) == action_position
          index = index_flat.reshape(self._states['board'].shape) == True
          if self._states['board'][index] != 0:
              print('Action:', action_position, (action_position // 9, action_position % 9),
                    'Illegal action: OCCUPIED!!!')
              if 0 in self._states['board']:
                  return TimeStep(StepType.MID,
                                  TicTacToeEnvironment.REWARD_ILLEGAL_MOVE,
                                  self._discount,
                                  self._states)
              else:
                  return TimeStep(StepType.LAST,
                                  TicTacToeEnvironment.REWARD_ILLEGAL_MOVE,
                                  self._discount,
                                  self._states)
          else:
              print('Action:', action_position, (action_position // 9, action_position % 9))
              self._states['board'][index] = action['value']
              is_final, reward = self._check_states(self._states['board'])
              if np.all(self._states['board'] == 0):
                  step_type = StepType.FIRST
              elif is_final:
                  step_type = StepType.LAST
              else:
                  step_type = StepType.MID
              return TimeStep(step_type, reward, self._discount, self._states)

  def _check_states(self, states: np.ndarray):
      """Check if the given states are final and calculate reward.
      Args:
        states: states of the board.
      Returns:
        A tuple of (is_final, reward) where is_final means whether the states
        are final are not, and reward is the reward for stepping into the states
        The meaning of reward: 0 = not decided or draw, 1 = win, -1 = loss
      """
      seqs = [
          # each row
          states[0, :],
          states[1, :],
          states[2, :],
          states[3, :],
          states[4, :],
          states[5, :],
          states[6, :],
          states[7, :],
          states[8, :],
          # each column
          states[:, 0],
          states[:, 1],
          states[:, 2],
          states[:, 3],
          states[:, 4],
          states[:, 5],
          states[:, 6],
          states[:, 7],
          states[:, 8],
          # incline L-R diagonal
          states[(0, 1, 2, 3), (5, 6, 7, 8)],
          states[(0, 1, 2, 3, 4), (4, 5, 6, 7, 8)],
          states[(0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)],
          states[(0, 1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7, 8)],
          states[(0, 1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 7, 8)],
          states[(0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7, 8)],
          states[(1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7)],
          states[(2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5, 6)],
          states[(3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5)],
          states[(4, 5, 6, 7, 8), (0, 1, 2, 3, 4)],
          states[(5, 6, 7, 8), (0, 1, 2, 3)],
          # incline R-L diagonal
          states[(3, 2, 1, 0), (0, 1, 2, 3)],
          states[(4, 3, 2, 1, 0), (0, 1, 2, 3, 4)],
          states[(5, 4, 3, 2, 1, 0), (0, 1, 2, 3, 4, 5)],
          states[(6, 5, 4, 3, 2, 1, 0), (0, 1, 2, 3, 4, 5, 6)],
          states[(7, 6, 5, 4, 3, 2, 1, 0), (0, 1, 2, 3, 4, 5, 6, 7)],
          states[(8, 7, 6, 5, 4, 3, 2, 1, 0), (0, 1, 2, 3, 4, 5, 6, 7, 8)],
          states[(8, 7, 6, 5, 4, 3, 2, 1), (1, 2, 3, 4, 5, 6, 7, 8)],
          states[(8, 7, 6, 5, 4, 3, 2), (2, 3, 4, 5, 6, 7, 8)],
          states[(8, 7, 6, 5, 4, 3), (3, 4, 5, 6, 7, 8)],
          states[(8, 7, 6, 5, 4), (4, 5, 6, 7, 8)],
          states[(8, 7, 6, 5), (5, 6, 7, 8)],
      ]

      """
      seqs_incline1 = np.array([
          states[(0, 1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 7, 8)],
          states[(1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7)],
          states[(7, 6, 5, 4, 3, 2, 1, 0), (0, 1, 2, 3, 4, 5, 6, 7)],
          states[(8, 7, 6, 5, 4, 3, 2, 1), (1, 2, 3, 4, 5, 6, 7, 8)],
      ])
      seqs_incline2 = np.array([
          states[(0, 1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7, 8)],
          states[(2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5, 6)],
          states[(6, 5, 4, 3, 2, 1, 0), (0, 1, 2, 3, 4, 5, 6)],
          states[(8, 7, 6, 5, 4, 3, 2), (2, 3, 4, 5, 6, 7, 8)],
      ])
      ... ...
      seqs = seqs.tolist()
     """

      X_win_str = ''.join(map(str, [1.0, 1.0, 1.0, 1.0]))
      O_win_str = ''.join(map(str, [2.0, 2.0, 2.0, 2.0]))

      for sublist in seqs:
          sublist_str = ''.join(map(str, sublist))
          if X_win_str in sublist_str:
              self.X_win = True
              return True, TicTacToeEnvironment.REWARD_WIN  # win
          elif O_win_str in sublist_str:
              self.O_win = True
              return True, TicTacToeEnvironment.REWARD_LOSS  # loss

      if 0 in states:
          return False, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL # Not final
      else:
          return True, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL  # draw