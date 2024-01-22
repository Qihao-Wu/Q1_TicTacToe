from IPython.display import clear_output
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from functools import partial
from Q1P2Env import TicTacToeEnvironment, print_tic_tac_toe, print_traj, ttt_action_fn, p2_reward_fn
from Q1P2Env import training_episode
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common
from tf_agents.trajectories.time_step import TimeStep
from DQN import IMAgent

# REWARD_ILLEGAL_MOVE = np.asarray(-5, dtype=np.float32)

################################################### Functions #######################################################
def collect_training_data():
    for game in range(episodes_per_iteration):
        training_episode(tf_ttt_env, player_1, player_2)

        p1_return = player_1.episode_return()
        p2_return = player_2.episode_return()

        if tf_ttt_env.envs[0].X_win:
            outcome = 'p1_win'
        elif tf_ttt_env.envs[0].O_win:
            outcome = 'p2_win'
        else:
            outcome = 'draw'

        games.append({
            'iteration': iteration,
            'game': game,
            'p1_return': p1_return,
            'p2_return': p2_return,
            'outcome': outcome,
            'final_step': tf_ttt_env.current_time_step()
        })

def train():
    for _ in range(train_steps_per_iteration):
        p1_train_info = player_1.train_iteration()
        p2_train_info = player_2.train_iteration()

        loss_infos.append({
            'iteration': iteration,
            'p1_loss': p1_train_info.loss.numpy(),
            'p2_loss': p2_train_info.loss.numpy()
        })

def plot_history():
    games_data = pd.DataFrame.from_records(games)
    loss_data = pd.DataFrame.from_records(loss_infos)
    loss_data['Player 1'] = np.log(loss_data.p1_loss)
    loss_data['Player 2'] = np.log(loss_data.p2_loss)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    loss_melted = pd.melt(loss_data,
                          id_vars=['iteration'],
                          value_vars=['Player 1', 'Player 2'])
    smoothing = iteration // 50
    loss_melted.iteration = smoothing * (loss_melted.iteration // smoothing)

    sns.lineplot(ax=axs[0][0],
                 x='iteration', hue='variable',
                 y='value', data=loss_melted)
    axs[0][0].set_title('Loss History')
    axs[0][0].set_ylabel('log-loss')

    returns_melted = pd.melt(games_data,
                             id_vars=['iteration'],
                             value_vars=['p1_return', 'p2_return'])
    returns_melted.iteration = smoothing * (returns_melted.iteration // smoothing)
    sns.lineplot(ax=axs[0][1],
                 x='iteration', hue='variable',
                 y='value', data=returns_melted)
    axs[0][1].set_title('Return History')
    axs[0][1].set_ylabel('return')

    games_data['p1_win'] = games_data.outcome == 'p1_win'
    games_data['p2_win'] = games_data.outcome == 'p2_win'
    games_data['draw'] = games_data.outcome == 'draw'
    grouped_games_data = games_data.groupby('iteration')
    cols = ['game', 'p1_win', 'p2_win', 'draw']
    grouped_games_data = grouped_games_data[cols]
    game_totals = grouped_games_data.max()['game'] + 1
    summed_games_data = grouped_games_data.sum()
    summed_games_data['p1_win_rate'] = summed_games_data.p1_win / game_totals
    summed_games_data['p2_win_rate'] = summed_games_data.p2_win / game_totals
    summed_games_data['draw_rate'] = summed_games_data.draw / game_totals
    summed_games_data['iteration'] = smoothing * (summed_games_data.index // smoothing)

    sns.lineplot(ax=axs[1][0],
                 x='iteration',
                 y='p1_win_rate',
                 data=summed_games_data,
                 label='Player 1 Win Rate')
    sns.lineplot(ax=axs[1][0],
                 x='iteration',
                 y='p2_win_rate',
                 data=summed_games_data,
                 label='Player 2 Win Rate')
    sns.lineplot(ax=axs[1][0],
                 x='iteration',
                 y='draw_rate',
                 data=summed_games_data,
                 label='Draw Ending Rate')
    axs[1][0].set_title('Outcomes History')
    axs[1][0].set_ylabel('ratio')

    plt.show()

def validate_agent(trained_agent, env, num_episodes):
    random_policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())

    def policy_step(policy, time_step):
        return policy.action(time_step).action

    win_count = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.is_last():
            agent_action = policy_step(trained_agent, time_step)
            time_step = env.step(agent_action)

            if time_step.is_last():
                # Check if the trained agent won the game
                if env.envs[0].X_win:  # Assuming 'X' is the trained agent
                    win_count += 1
                break

            opponent_action = policy_step(random_policy, time_step)
            time_step = env.step(opponent_action)

    win_rate = win_count / num_episodes
    return win_rate
#####################################################################################################################

tic_tac_toe_env = TicTacToeEnvironment()
tf_ttt_env = TFPyEnvironment(tic_tac_toe_env)

num_iterations = 1000
initial_collect_episodes = 100
episodes_per_iteration = 10
train_steps_per_iteration = 1
training_batch_size = 512
training_num_steps = 2
replay_buffer_size = 3 * episodes_per_iteration * 9
learning_rate = 1e-3
plot_interval = 50

iteration = 1
games = []
loss_infos = []

player_1 = IMAgent(
    tf_ttt_env,
    action_spec = tf_ttt_env.action_spec()['position'],
    action_fn = partial(ttt_action_fn, 1),
    name='Player1',
    learning_rate = learning_rate,
    training_batch_size = training_batch_size,
    training_num_steps = training_num_steps,
    replay_buffer_max_length = replay_buffer_size,
    td_errors_loss_fn=common.element_wise_squared_loss
)
player_2 = IMAgent(
    tf_ttt_env,
    action_spec = tf_ttt_env.action_spec()['position'],
    action_fn = partial(ttt_action_fn, 2),
    reward_fn = p2_reward_fn,
    name='Player2',
    learning_rate = learning_rate,
    training_batch_size = training_batch_size,
    training_num_steps = training_num_steps,
    replay_buffer_max_length = replay_buffer_size,
    td_errors_loss_fn=common.element_wise_squared_loss
)

print('Collecting Initial Training Sample...')
for _ in range(initial_collect_episodes):
    training_episode(tf_ttt_env, player_1, player_2)
print('===== Samples Collected =====')

try:
    if iteration > 1:
        plot_history()
        clear_output(wait=True)
    while iteration < num_iterations:
        collect_training_data()
        train()
        iteration += 1
        if iteration % plot_interval == 0:
            plot_history()
            clear_output(wait=True)
        print('Iteration Completed...')

except KeyboardInterrupt:
    clear_output(wait=True)
    print('Interrupting training, plotting history...')
    plot_history()

#####################################################################################################################
# Save trained models
# After training, create a PolicySaver
policy_saver_1 = PolicySaver(player_1.policy)
policy_saver_2 = PolicySaver(player_2.policy)
# Save the policy to a directory. save_path = '/path/to/save/your/policy'
policy_saver_1.save('trained_player_1')
policy_saver_2.save('trained_player_2')
#####################################################################################################################

# # Validation
# reset an untrained agent, player_2t, as a random policy agent to validate the training.
player_2t = IMAgent(
    tf_ttt_env,
    action_spec = tf_ttt_env.action_spec()['position'],
    action_fn = partial(ttt_action_fn, 2),
    reward_fn = p2_reward_fn,
    name='Player2t'
    # learning_rate = learning_rate,
    # training_batch_size = training_batch_size,
    # training_num_steps = training_num_steps,
    # replay_buffer_max_length = replay_buffer_size,
    # td_errors_loss_fn=common.element_wise_squared_loss
)
validation = []
# generate a 100 instances of trained v.s untrained for validation
for game in range(100):
    training_episode(tf_ttt_env, player_1, player_2t)
    p1_return = player_1.episode_return()
    p2t_return = player_2t.episode_return()
    if tf_ttt_env.envs[0].X_win:
        outcome = 'p1_win'
    elif tf_ttt_env.envs[0].O_win:
        outcome = 'p2t_win'
    else:
        outcome = 'draw'
    validation.append({
        'game': game,
        'p1_return': p1_return,
        'p2t_return': p2t_return,
        'outcome': outcome,
    })

print('Validation on 100 plays with random policy:', validation)
pd.DataFrame([validation]).to_csv('validation.csv')
