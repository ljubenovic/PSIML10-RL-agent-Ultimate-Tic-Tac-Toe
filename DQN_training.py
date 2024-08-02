from envs.env_two_player import TwoPlayerEnv
from agents.agent_dqn import DQNAgent

if __name__ == '__main__':
    
    env = TwoPlayerEnv()

    epsilon = 0.2
    n_episodes = 50000
    n_save = 500

    name = '_train'
    agent = DQNAgent(env, epsilon=epsilon, loading=False, masked = True, n_episodes = n_episodes, n_save = n_save, name = name)
