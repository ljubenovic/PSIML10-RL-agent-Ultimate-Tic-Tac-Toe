import os
from datetime import datetime
from envs.env_two_player import TwoPlayerEnv
from dqn_agent.agent_dqn import DQNAgent

if __name__ == '__main__':
    
    env = TwoPlayerEnv()

    # Creating a folder to save the results
    now = datetime.now()
    results_folder = os.path.join("results", now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(results_folder, exist_ok=True)

    # DQN parameters
    epsilon = 0.3
    name = '_train'
    """
    model_flag:
        0 - Load the model with the name "name"
        1 - Train the model using self-play
        2 - Train the model to play as X against a random agent
    """
    model_flag = 2
    masked = True   # Selecting an action only from the set of valid actions
    target_update = 100

    # Training parameters
    n_episodes = 500     # Number of episodes for training
    n_save = 100          # saving the trained model every n_save episodes

    agent = DQNAgent(env, epsilon, model_flag, masked, n_episodes, n_save, name, target_update, results_folder)
