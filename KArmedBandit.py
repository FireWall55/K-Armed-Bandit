import random
from matplotlib import pyplot as plt 
import numpy as np

class Environment:
    def __init__(self):
        self.time = 1

    def admissible_actions(self):
        return list(range(1,5))

    def get_reward(self, action):
        self.time += 1
        return random.gauss(action*2+2,1)


class Agent:
    def __init__(self):
        self.current_reward = 0.0

    def step(self, env, q, n):
        actions = env.admissible_actions()
        reward = 0
        if random.randint(1,10) == 10:
            action = random.choice(actions)
            reward = env.get_reward(action)
            n[action-1] += 1
            q[action-1] += 1/n[action-1] * (reward - q[action-1])
        else:
            maxValue = max(q) #gets max value from q's
            possible_actions = []
            for num in range(len(q)):
                if maxValue == q[num]:
                    possible_actions.append(num+1)
            #all possible actions set
            if len(possible_actions) != 0:
                action = random.choice(possible_actions)
                reward = env.get_reward(action)
                n[action-1] += 1
                q[action-1] += 1/n[action-1] * (reward - q[action-1])
        self.current_reward = reward
        
        
def rewards_hist(T = 10):
    env = Environment()
    agent = Agent()
    rewards = []
    q = [0,0,0,0]
    n = [0,0,0,0]
    while env.time <= T:
        agent.step(env, q, n)
        rewards.append(agent.current_reward)
        # print("q: ",q)
        # print("n: ",n)
    return np.asarray(rewards)


from matplotlib import pyplot as plt 
import numpy as np   
T = 100
plt.figure(figsize=(16,6))
plt.plot(range(1,T+1),rewards_hist(T), linestyle='--', marker='o', markersize=4, label='random actions')
plt.xlabel("t")
plt.ylabel("$R_t$")
plt.title('Dependence of historical rewards $R_t$ on $t$')
plt.grid(True)
plt.legend(loc = 'upper right')
plt.show()