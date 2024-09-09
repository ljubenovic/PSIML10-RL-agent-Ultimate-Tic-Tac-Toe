from random_agent import RandomAgent

def DQN_vs_random(agent_DQN, env, n_episodes, is_DQN_first = True):

    agent_random = RandomAgent()
    if is_DQN_first:
        random_turn = 2
    else:
        random_turn = 1

    results = [0,0,0]
    for _ in range(n_episodes):

        obs = env.reset()
        game = True

        currentPlayer = 1
        flag_x = 1
        
        while game:
            if currentPlayer == random_turn: # Random agent
                action = agent_random.getAction(env) # Take a random action
                obs, reward, done, _ = env.step(action)
                if done == True:
                    game = False
            else: # DQN agent
                action = agent_DQN.getAction(env, obs, True, is_DQN_first)
                if action < 0:  # If the aciton is negative this means that the agent asks to close the game
                    done = True
                elif action < 81:   # Otherwise, if the action is valid we play it in the env
                    obs, reward, done, _ = env.step(action)
                    if reward == -100:
                        print('ERROR!')
                if done == True:
                    game = False
            currentPlayer = 3-currentPlayer
            flag_x = 1-flag_x

        if env.pygame.board.state == random_turn:
            results[1] += 1
        elif env.pygame.board.state < 3:
            results[0] += 1
        else:
            results[2] += 1
    
    env.close()

    return results

