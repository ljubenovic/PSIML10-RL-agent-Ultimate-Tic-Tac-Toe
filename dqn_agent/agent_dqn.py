import numpy as np
import torch
import tqdm

from envs.env_two_player import TwoPlayerEnv
from agent import Agent
from dqn_agent.deep_q_network import DQNetwork
from dqn_agent.replay_buffer import ReplayBuffer
from random_agent import RandomAgent
from dqn_vs_random import DQN_vs_random

from utils import save_dqn_parameters, save_loss, save_loss_graph, save_results_against_random


def processObs(observation):
    return np.array(observation[0] + observation[1] +[1 if i in observation[2] else 0 for i in range(9)]).flatten()

class DQNAgent(Agent):

    def __init__(self, env, eps, model_flag=0, masked = True, n_episodes = 1000, n_save = 500, name = "_train", target_update=100, results_folder=""):

        lr = 1e-6
        gamma = 0.99
        batch_size = 128
        state_len = 99  # (grid, largeGrid, possible)
        n_actions = env.action_space.n  # 81
        mem_size = 1000000
        min_mem_size = batch_size
        eps_min = 0.01
        n_step = 5

        super().__init__(self)
        self.it_counter = 0            # how many timesteps have passed already
        self.gamma = gamma             # gamma hyperparameter
        self.batch_size = batch_size   # batch size hyperparameter for neural network
        self.state_len = state_len     # how long the state vector is
        self.n_actions = n_actions     # number of actions the agent can take
        self.eps_start = eps           # epsilon start value (1=completly random)
        self.eps = eps
        self.eps_min = eps_min # the minimum value
        self.mem_size = mem_size
        self.min_mem_size = min_mem_size
        self.n_step = n_step
        self.target_update = target_update
        self.results_folder = results_folder

        save_dqn_parameters(self.results_folder, lr, gamma, eps, eps_min, batch_size, mem_size, min_mem_size, n_step,target_update,model_flag)

        output_range = (0,1) if (model_flag % 2 == 0) else (-1,1)
        self.q = DQNetwork(state_len, n_actions, lr, output_range)

        # target network
        self.q_target = DQNetwork(state_len, 81, lr, output_range)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

         # memory for 1-step Learning
        self.memory = ReplayBuffer(state_len, mem_size, batch_size, n_step=1, gamma=gamma)
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(state_len, mem_size, batch_size, n_step=n_step, gamma=gamma)
        
        if model_flag == 0:
            self.q.load(name, self.results_folder)
        elif model_flag == 1:
            self.learnNN(env, masked, n_episodes, n_save, name)
        elif model_flag == 2:
            self.learnNN_as_X(env, masked, n_episodes, n_save, name)


    def getAction(self, env, observation, check_validity, flag_x):
        
        observation = torch.tensor(processObs(observation), dtype = torch.float32).to(self.q.device)
        with torch.no_grad():
            q = self.q(observation)
        if flag_x:
            action = int(torch.argmax(q))
        else:
            action = int(torch.argmin(q))

        if check_validity:
            valid_actions = env.valid_actions()
            if action in valid_actions:
                pass
            else:
                q_min = torch.min(q)
                mask = torch.tensor([True if i in valid_actions else False for i in range(env.action_space.n)]).to(self.q.device)
                new_q = (q.detach() - q_min + 1.) *  mask
                if flag_x:
                    action = int(torch.argmax(new_q))
                else:
                    masked_q = torch.where(new_q != 0, new_q, float('inf'))
                    action = int(torch.argmin(masked_q))

        return action

    def pickActionMaybeRandom(self, env, observation, check_validity, flag_x):
        if np.random.random() < self.eps:
            valid_actions = env.valid_actions()
            return int(np.random.choice(valid_actions))
        else:
            return self.getAction(env, observation, check_validity, flag_x)


    def compute_dqn_loss(self, samples, gamma):
        """Return dqn loss."""
        device = self.q.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].astype(np.int32).reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        with torch.no_grad():
            next_q_value = self.q_target(next_state).max(dim=1, keepdim=True)[0].detach()
            target = (reward + gamma * next_q_value * (1-done)).to(self.q.device)
        curr_q_value = self.q(state).gather(1, action)  # prediction
        # calculate dqn loss
        #loss = torch.nn.functional.smooth_l1_loss(curr_q_value, target, reduction='mean')
        loss = self.q.loss(curr_q_value, target)

        return loss

    def learn(self):
        
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self.compute_dqn_loss(samples, self.gamma)
        if self.use_n_step:
            samples = self.memory_n.sample_batch_from_idxs(indices)
            gamma = self.gamma ** self.n_step
            n_loss = self.compute_dqn_loss(samples, gamma)
            loss += n_loss

        self.q.optimizer.zero_grad()
        loss.backward()
        self.q.optimizer.step()

        """def get_current_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']"""

        self.it_counter += 1
        
        return loss.item()
        

    def linear_schedule(self, n_episodes, episode):
        slope = (self.eps_min - self.eps_start) / n_episodes
        eps = max(slope * episode + self.eps_start, self.eps_min)
        return eps
    

    def learnNN(self, env, masked = True, n_episodes = 1000, n_save = 500, trainingName = "", results_folder = ""):
   
        loss_arr = []
        loss = []
        lr_arr = []

        for episode in tqdm.tqdm(range(n_episodes)):
            
            state = env.reset()
            score = 0
            done = 0
            self.eps = self.linear_schedule(n_episodes, episode)

            flag_x = 1
            while not done: # while the episode is not over yet
                
                action = None
                
                if masked:
                    action = self.pickActionMaybeRandom(env, state, True, flag_x)

                new_state, reward, done, _ = env.step(action, flag_x) # performing the action in the environment
                
                
                if flag_x:  # X plays
                    pass
                else:   # O plays
                    reward = -1*reward
                
                score += reward #  the total score during this round

                self.replay_buffer.store_transition(processObs(state), action, reward, processObs(new_state), done, flag_x)
                
                (loss_tmp, lr) = self.learn()   # the agent learns after each timestep
                lr_arr.append(lr)

                if loss_tmp:
                    loss_arr.append(loss_tmp)
                state = new_state
                flag_x = 1-flag_x
                
            if ((episode+1) % 50 == 0) and episode >= self.min_mem_size:
                loss = save_loss(loss, loss_arr, self.results_folder)
                loss_arr = []

            if (episode+1) % n_save == 0:

                name = trainingName + "_" + str(episode+1)
                self.q.save(name, self.results_folder)

                env = TwoPlayerEnv()
                agent_DQN = DQNAgent(env, eps=0, model_flag=0, name=name, results_folder=self.results_folder)

                results1 = DQN_vs_random(agent_DQN, env, n_episodes=1000, is_DQN_first = True)
                print('results = ', results1, ' (DQN first)')
                results2 = DQN_vs_random(agent_DQN, env, n_episodes=1000, is_DQN_first = False)
                print('results = ', results2, ' (Random first)')

                save_results_against_random(self.results_folder, episode, results1, results2)
                save_loss_graph(self.results_folder, loss, episode)
                
        env.close()
        self.q.save(trainingName + "_final", self.results_folder)
        

    def learnNN_as_X(self, env, masked = True, n_episodes = 1000, n_save = 500, trainingName = "", results_folder = ""):

        loss_arr = []
        loss = []
        #lr_arr = []
        agent_random = RandomAgent()
        update_cnt = 0

        for episode in tqdm.tqdm(range(n_episodes)):

            state = env.reset()
            done = 0
            self.eps = self.linear_schedule(n_episodes, episode)

            while not done:

                action = None
                
                # DQN plays as X
                action = self.getAction(env, state, True, 1)
                # step
                new_state, reward, done, _ = env.step(action)
                self.q.transition = [processObs(state), action, reward, processObs(new_state), done]
                if self.use_n_step:
                    one_step_transition = self.memory_n.store_transition(*self.q.transition)
                else:
                    one_step_transition = self.q.transition
                if one_step_transition:
                    self.memory.store_transition(*one_step_transition)

                if len(self.memory) >= self.batch_size: # if training is ready
                    loss_tmp = self.learn()
                    if loss_tmp:
                        loss_arr.append(loss_tmp)
                    update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

                state = new_state
                if done == True:
                    game = False
                    break
                
                # Random plays as O
                action = agent_random.getAction(env)
                new_state, reward, done, _ = env.step(action)
                #reward = -1*reward
                if done == True:
                    game = False
                    break

                state = new_state
                
            if ((episode+1) % 50 == 0) and episode >= self.min_mem_size:
                loss = save_loss(loss, loss_arr, self.results_folder)
                loss_arr = []
                

            if (episode+1) % n_save == 0:

                name = trainingName + "_" + str(episode+1)
                self.q.save(name, self.results_folder)

                env = TwoPlayerEnv()
                agent_DQN = DQNAgent(env, eps=0, model_flag=0, name=name, results_folder=self.results_folder)

                results = DQN_vs_random(agent_DQN, env, n_episodes=1000, is_DQN_first = True)
                print('results = ', results, ' (DQN first)')

                save_results_against_random(self.results_folder, episode, results)
                save_loss_graph(self.results_folder, loss, episode)
  
        env.close()
        self.q.save(trainingName + "_final_X", self.results_folder)

        return

