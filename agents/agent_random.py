from agents.agent import Agent

class Agent:

    def __init__(self, player = 1):
        self.player = player

    def getAction(self, env, observation):
        return 0

class RandomAgent(Agent):

    def __init__(self, player = 1):
        super().__init__(self)

    def getAction(self, env):
        action = env.action_space.sample()
        return action