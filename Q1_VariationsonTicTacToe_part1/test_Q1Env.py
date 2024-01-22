import unittest
import random
from functools import partial
from itertools import cycle

import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import StepType
from Q1Env import TicTacToeEnvironment, print_tic_tac_toe, ttt_action_fn, p2_reward_fn
from DQN import IMAgent

class TestTicTacToeEnvironment(unittest.TestCase):

############################## Unit tests ##############################

    def setUp(self):
        # Initialize the environment before each test
        self.env = TicTacToeEnvironment()

    def test_reset(self):
        # Call the reset method
        time_step = self.env.reset()
        # Check if the initial observation is as expected (empty board)
        expected_board = np.zeros((9, 9), dtype=np.int32)
        self.assertTrue((time_step.observation == expected_board).all(),
                        "The board should be empty after reset.")
        # Check if the step type is the first step
        self.assertEqual(time_step.step_type, StepType.FIRST, "The step type should be `StepType.FIRST` after reset.")

    def test_observation_spec(self):
        # The observation_spec method should return the spec of the observation space
        observation_spec = self.env.observation_spec()
        self.assertIsInstance(observation_spec, BoundedArraySpec,
                              "The board spec should be a BoundedArraySpec instance.")
        self.assertEqual(observation_spec.shape, (9, 9), "The board should be 9x9.")
        self.assertEqual(observation_spec.dtype, np.int32, "The board should have dtype int32.")

    def test_action_spec(self):
        # The action_spec method should return the spec of the action space
        action_spec = self.env.action_spec()
        self.assertIsInstance(action_spec, dict, "The action spec should be a dictionary.")
        # Assuming the action is a dict with 'position' and 'value'
        self.assertIn('position', action_spec, "The action spec should include the position.")
        self.assertIn('value', action_spec, "The action spec should include the value.")
        # Test the specifics of each action component
        self.assertIsInstance(action_spec['position'], BoundedArraySpec,
                              "The position spec should be a BoundedArraySpec instance.")
        self.assertIsInstance(action_spec['value'], BoundedArraySpec,
                              "The value spec should be a BoundedArraySpec instance.")
        # Assuming position is a single integer and value is either 1 or 2
        self.assertEqual(action_spec['position'].shape, (), "Position spec should be a scalar.")
        self.assertEqual(action_spec['position'].dtype, np.int32, "Position should have dtype int32.")
        self.assertEqual(action_spec['value'].dtype, np.int32, "Value should have dtype int32.")
        # Add checks for bounds if applicable
        self.assertTrue((action_spec['position'].minimum >= 0) and (action_spec['position'].maximum <= 80),
                        "Position should be within board limits.")
        self.assertTrue((action_spec['value'].minimum == 1) and (action_spec['value'].maximum == 2),
                        "Value should be 1 or 2.")

    def test_get_state(self):
        # The get_state method should return the current state of the environment
        self.env.reset()
        state = self.env.get_state()
        self.assertTrue(hasattr(state, 'observation'), "The state should have an observation.")
        self.assertTrue(hasattr(state, 'step_type'), "The state should have a step_type.")
        self.assertTrue(hasattr(state, 'reward'), "The state should have a reward.")
        self.assertTrue(hasattr(state, 'discount'), "The state should have a discount.")
        # You may also test the initial values of these fields
        self.assertTrue((state.observation == np.zeros((9, 9))).all(), "Initial board should be all zeros.")
        self.assertEqual(state.step_type, StepType.FIRST, "The initial step type should be `StepType.FIRST`.")
        self.assertTrue(state.reward==0, "Initial reward should be 0.")
        self.assertEqual(state.discount, 1.0, "Initial discount should be 1.0.")

    def test_legal_actions(self):
        # Assuming `_legal_actions` is made public for testing purposes
        self.env.reset()
        legal_actions = self.env._legal_actions(self.env.get_state().observation)
        self.assertEqual(len(legal_actions), 81, "All positions should be legal at the start of the game.")

    def test_illegal_move_outside(self):
        self.env.reset()
        state_board = self.env._states
        self.env.step({'position': 100, 'value': 1})  # step method exists and 100 is always out of bounds
        self.assertTrue((state_board == self.env._states).all(),
                        "After an outside invalid step, the board should not change.")

############################## Regression tests ##############################

    def test_regression_illegal_move_occupied(self):
        self.env.reset()
        # fill all the board states with value, 1 or 2, to create an occupation
        self.env._states[:] = 1  # 2
        # step one valid action with a different value, 2:
        self.env.step({'position': 60, 'value': 2})
        self.assertTrue(self.env._states.all() != 2,
                        "The value 2 cannot be placed due to all board states are occpuied.")

    def test_regression_end_of_episode(self):
        self.env.reset()
        # fill all the board states with value, 1 or 2, to create an end.
        # This can be tested with any 9x9 board placement inputs.
        self.env._states[:] = 1  # 2
        self.assertTrue(self.env._check_states(self.env._states),
                                               "a row/column/diagonal of >=4 places and no empty must end the game.")

############################## Integration tests ##############################

    def test_integration_random_trial(self):
        ts = self.env.reset()
        print('Reward:', ts.reward, 'Board:')
        print_tic_tac_toe(ts.observation)
        random.seed(1)
        player = 1
        while not ts.is_last():
            action = {
                'position': np.asarray(random.randint(0, 80)),
                'value': player
            }
            ts = self.env.step(action)
            print('Player:', player, 'Action:', action['position'],
                  'Reward:', ts.reward, 'Board:')
            print_tic_tac_toe(ts.observation)
            player = 1 + player % 2
        self.assertTrue(ts.is_last(), "The random trial game ends with visualizations.")

    def test_integration_agent_trial(self):
        # use two untrained initial agents to play the game. It is equivalent to the random policy
        tf_ttt_env = TFPyEnvironment(TicTacToeEnvironment())
        player_1 = IMAgent(
            tf_ttt_env,
            action_spec=tf_ttt_env.action_spec()['position'],
            action_fn=partial(ttt_action_fn, 1),
            name='Player1'
        )
        player_2 = IMAgent(
            tf_ttt_env,
            action_spec=tf_ttt_env.action_spec()['position'],
            action_fn=partial(ttt_action_fn, 2),
            reward_fn=p2_reward_fn,
            name='Player2'
        )
        ts = tf_ttt_env.reset()
        # arbitrary starting point to add variety
        random.seed(1)
        start_player_id = random.randint(1, 2)
        tf_ttt_env.step({
            'position': tf.convert_to_tensor([random.randint(0, 80)]),
            'value': start_player_id
        })
        ts = tf_ttt_env.current_time_step()
        print('Random start board:')
        print_tic_tac_toe(ts.observation.numpy())
        if start_player_id == 2:
            players = cycle([player_1, player_2])
        else:
            players = cycle([player_2, player_1])
        while not ts.is_last():
            player = next(players)
            print(f'Player: {player.name}')
            player.act(collect=True)
            ts = tf_ttt_env.current_time_step()
            print(f'Reward: {ts.reward[0]}')
            print_tic_tac_toe(ts.observation.numpy())
        self.assertTrue(ts.is_last(), "The initial agents game ends with visualizations.")

'''
############## Visualize replay buffer #######################
for _ in range(10):
    training_episode(tf_ttt_env, player_1, player_2)

print('Number of trajectories recorded by P1:',
      player_1._replay_buffer.num_frames().numpy())
print('Number of trajectories recorded by P2:',
      player_2._replay_buffer.num_frames().numpy())

tf.random.set_seed(5)
traj_batches, info = player_1._replay_buffer.get_next(num_steps=2, sample_batch_size=3)

for i in range(3):
    action = traj_batches.action[i, 0].numpy()
    print('Action: Place \'X\' at', (action // 9, action % 9))
    reward = traj_batches.reward[i, 0].numpy()
    print('Reward:', reward)
    print_traj(traj_batches.observation[i])
    print()
'''

# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()