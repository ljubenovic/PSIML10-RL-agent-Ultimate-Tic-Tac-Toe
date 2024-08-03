from html.entities import name2codepoint
from envs.env_two_player import TwoPlayerEnv
from agents.agent_random import RandomAgent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pygame
import numpy as np
import torch
import tqdm
import csv

## agent

class Agent:

    def __init__(self, player = 1):
        self.player = player

    def getAction(self, env, observation):
        return 0

def processObs(observation):
    return np.array(observation[0] + observation[1] +[1 if i in observation[2] else 0 for i in range(9)]).flatten()

class ReplayBuffer(object):
    def __init__(self, state_len, mem_size):
        
        self.state_len = state_len
        self.mem_size = mem_size
        self.mem_counter = 0
        self.states = np.zeros((mem_size, state_len), dtype=np.float32)
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.new_states = np.zeros((mem_size, state_len), dtype=np.float32)
        self.dones = np.zeros(mem_size, dtype=np.int32)
        self.flags = np.zeros(mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done, flag_x):
        index = self.mem_counter%self.mem_size
        self.states[index, :] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index, :] = new_state
        self.dones[index] = done
        self.mem_counter += 1
        self.flags[index] = flag_x

    def sample_memory(self, batch_size):
        max_memory = min(self.mem_size, self.mem_counter)
        # Nasumicno uzorkovanje tranzicija iz memorije (zbog korelisanosti izmedju susednih tranzicija)
        batch = np.random.choice(np.arange(max_memory), batch_size, replace=False)
        states = self.states[batch, :]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch, :]
        dones = self.dones[batch]
        flags = self.flags[batch]
        return states, actions, rewards, new_states, dones, flags


class DQNetwork(torch.nn.Module):
    def __init__(self, state_len, n_actions, learning_rate):
        super(DQNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_len, 81),
            torch.nn.ReLU(),
            torch.nn.Linear(81, 81),
            torch.nn.ReLU(),
            torch.nn.Linear(81, n_actions),
            torch.nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.to(self.device)

    def forward(self,state):
        return self.network(state)

    def save(self, name = ""):
        torch.save(self,os.path.dirname(__file__)+"/DQNetwork" + name + ".pt")

    def load(self, name = ""):
        self = torch.load(os.path.dirname(__file__)+"/DQNetwork" + name + ".pt")
        self.eval()


class DQNAgent(Agent):

    def __init__(self, env, epsilon, loading=True, masked = True, n_episodes = 1000, n_save = 500, name = "_train"):

        learning_rate = 1e-5
        gamma = 0.99
        batch_size = 128
        state_len = 99  # (grid, largeGrid, possible)
        n_actions = env.action_space.n  # 81
        mem_size = 1000000
        min_memory_for_training = batch_size
        
        epsilon = epsilon  # 0.3
        epsilon_dec = 0.998
        epsilon_min = 0.1
        #frozen_iterations = 6

        super().__init__(self)
        self.it_counter = 0            # how many timesteps have passed already
        self.gamma = gamma             # gamma hyperparameter
        self.batch_size = batch_size   # batch size hyperparameter for neural network
        self.state_len = state_len     # how long the state vector is
        self.n_actions = n_actions     # number of actions the agent can take
        self.epsilon_start = epsilon   # epsilon start value (1=completly random)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min # the minimum value
        self.mem_size = mem_size

        with open('parameters.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['learning rate',learning_rate])
            csvwriter.writerow(['gamma',0.99])
            csvwriter.writerow(['epsilon',epsilon])
            csvwriter.writerow(['epsilon dec',epsilon_dec])
            csvwriter.writerow(['epsilon min',epsilon_min])
            csvwriter.writerow(['batch size',batch_size])
            csvwriter.writerow(['mem size',mem_size])
            csvwriter.writerow(['min memory for training',min_memory_for_training])

        self.min_memory_for_training = min_memory_for_training
        self.q = DQNetwork(state_len, 81, learning_rate)
        self.replay_buffer = ReplayBuffer(self.state_len, mem_size)

        if loading :
            self.q.load(name)
        else :
            self.learnNN(env, masked, n_episodes, n_save, name)

    def getAction(self, env, observation, check_validity, flag_x):
        observation = torch.tensor(processObs(observation), dtype = torch.float32).to(self.q.device)
        q = self.q.forward(observation)
        if flag_x:
            action = int(torch.argmax(q))
        else:
            action = int(torch.argmin(q))

        if check_validity: # checks for action validity

            valid_actions = env.valid_actions()

            if action in valid_actions:
                pass
            else:
                q_min = float(torch.min(q))
                mask = torch.tensor([True if i in valid_actions else False for i in range(env.action_space.n)])
                new_q = (q.detach() - q_min + 1.) *  mask
                if flag_x:
                    action = int(torch.argmax(new_q))
                else:
                    masked_q = torch.where(new_q != 0, new_q, float('inf'))
                    action = int(torch.argmin(masked_q))

        return action

    def pickActionMaybeRandom(self, env, observation, check_validity, flag_x):
        if np.random.random() < self.epsilon:
            # nasumicna akcija
            valid_actions = env.valid_actions()
            return int(np.random.choice(valid_actions))
        else:
            # akcija koja maksimizira Q vrednost
            return self.getAction(env, observation, check_validity, flag_x)

    def learn(self, error):
        
        # proverava se da li je prikupljeno dovoljno iskustva da se zapocne trening
        if self.replay_buffer.mem_counter < self.min_memory_for_training:
            return
        # uzorkovanje nasumicnih iskustava iz buffer-a
        states, actions, rewards, new_states, dones, flags = self.replay_buffer.sample_memory(self.batch_size)
        
        self.q.optimizer.zero_grad()
        states_batch = torch.tensor(states, dtype = torch.float32).to(self.q.device)
        new_states_batch = torch.tensor(new_states, dtype = torch.float32).to(self.q.device)
        actions_batch = torch.tensor(actions, dtype = torch.long).to(self.q.device)
        rewards_batch = torch.tensor(rewards, dtype = torch.float32).to(self.q.device)
        dones_batch = torch.tensor(dones, dtype = torch.float32).to(self.q.device)
        flags_batch = torch.tensor(flags, dtype= torch.bool).to(self.q.device)

        # Bellmmanova jednacina
        with torch.no_grad():
            q_values = self.q(new_states_batch)
            max_q_values = q_values.max(axis=1).values
            min_q_values = q_values.min(axis=1).values
            chosen_q_values = torch.where(flags_batch, min_q_values, max_q_values)
            target = rewards_batch + torch.mul(self.gamma * chosen_q_values, (1 - dones_batch))

        # Estimacija
        prediction = self.q.forward(states_batch).gather(1,actions_batch.unsqueeze(1)).squeeze(1)
        
        loss = self.q.loss(prediction, target) # TD error
        loss.backward()  # Compute gradients
        self.q.optimizer.step()  # Backpropagate error

        self.it_counter += 1
        
        return loss.item()

    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
    
    def learnNN(self, env, masked = True, n_episodes = 1000, n_save = 500, trainingName = ""):
        l_epsilon = []
        l_win = []
        loss_arr = []
        loss = []
        sum_win = 0

        output_dir = 'loss_graphs'
        
        for episode in tqdm.tqdm(range(n_episodes)):
            
            state = env.reset() # resetting the environment after each episode
            score = 0
            done = 0
            
            flag_x = ((episode % 2)==0)
            self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_min, n_episodes, episode)
            
            while not done: # while the episode is not over yet
                action = None
                if masked:
                    action = self.pickActionMaybeRandom(env, state, True, flag_x)

                new_state, reward, done, error = env.step(action) # performing the action in the environment
                
                if flag_x:
                    pass
                else:
                    reward = -1*reward
                
                score += reward #  the total score during this round

                self.replay_buffer.store_transition(processObs(state), action, reward, processObs(new_state), done, flag_x)   # store timestep for experiene replay
                loss_tmp = self.learn(error)   # the agent learns after each timestep
                
                if loss_tmp:
                    loss_arr.append(loss_tmp)
                state = new_state

            if env.pygame.board.state == 1:
                sum_win +=1
            elif env.pygame.board.state == 2:
                sum_win -= 1
            elif env.pygame.board.state == 3:
                sum_win += 0

            l_epsilon.append(self.epsilon)
            l_win.append(sum_win)

            if ((episode+1) % 50 == 0) and episode >= self.min_memory_for_training:
                
                loss_avg = torch.mean(torch.tensor(loss_arr))
                with open('loss.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(loss_arr)
                loss_arr = []
                loss.append(loss_avg.item())
                print('\n',loss_avg)

            if (episode+1) % n_save == 0:

                # Cuvanje istreniranog modela
                name = trainingName + "_" + str(episode+1)
                self.q.save(name)

                # Evaluacija performansi protiv random igraca
                env = TwoPlayerEnv()
                agent_DQN = DQNAgent(env, epsilon=0, loading=True, name=name)

                file_path = 'results_train.txt'
                results1 = DQN_vs_random(agent_DQN, env, n_episodes=1000, is_DQN_first = True, file_path = file_path)
                print('results = ', results1, ' (DQN first)')
                results2 = DQN_vs_random(agent_DQN, env, n_episodes=1000, is_DQN_first = False, file_path = file_path)
                print('results = ', results2, ' (Random first)')

                # Cuvanje rezultata igranja sa random-om
                with open('results.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([1,*results1])
                    csvwriter.writerow([2,*results2])

                # Cuvanje grafika loss funkcije  
                plt.figure()
                plt.plot(loss)
                plt.xlabel('# episodes [x 50]')
                plt.ylabel('loss')
                plt.title(f'Average loss')
                plt.savefig(os.path.join(output_dir, f'loss_epoch_{episode + 1}.png'))
                plt.close()
                
        env.close()
        self.q.save(trainingName + "_final")
        print(l_epsilon)
        print("\n")
        print(l_win)

def DQN_vs_random(agent_DQN, env, n_episodes, is_DQN_first = True, file_path = 'results.txt'):

    agent_random = RandomAgent()
    if is_DQN_first:
        random_turn = 2
    else:
        random_turn = 1

    results = [0, 0, 0]

    for _ in range(n_episodes):

        obs = env.reset()

        game = True
        while game:

            if env.pygame.board.currentPlayer == random_turn: # Random agent
                
                action = agent_random.getAction(env) # Take a random action
                obs, reward, done, _ = env.step(action)
                if done == True:
                    game = False

            else: # DQN agent
                action = agent_DQN.getAction(env, obs, True, is_DQN_first)

                if action < 0:  # If the aciton is negative this means that the agent asks to close the game
                    done = True

                elif action < 81:   # Otherwise, if the action is valid we play it in the env
                    obs, reward, done, info = env.step(action)
                    if reward == -100:
                        print('ERROR!')
                if done == True:
                    game = False

        if env.pygame.board.state == random_turn:
            results[1] += 1
        elif env.pygame.board.state < 3:
            results[0] += 1
        else:
            results[2] += 1
    
    if is_DQN_first:
        with open(file_path, 'a') as file:
            file.write('DQN played first. Random played second.\n')
            file.write('DQN won in {} games\n'.format(results[0]))
            file.write('Random won in {} games\n'.format(results[1]))
            file.write('Draw: {} games\n'.format(results[2]))
            file.write('\n')
    else:
        with open(file_path, 'a') as file:
            file.write('Random played first. DQN played second.\n')
            file.write('DQN won in {} games\n'.format(results[0]))
            file.write('Random won in {} games\n'.format(results[1]))
            file.write('Draw: {} games\n'.format(results[2]))
            file.write('\n')

    env.close()

    return results



if __name__ == '__main__':

    env = TwoPlayerEnv()

    name = "_test_3000"
    agent_DQN = DQNAgent(env, epsilon=0, loading=True, name=name)

    results = DQN_vs_random(agent_DQN, env, n_episodes=1000, is_DQN_first = True, file_path = 'results_0.txt')
    print('results = ', results)
