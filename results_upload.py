import gym
import api_key
from qlearn.qlearningNN import qneuralagent

env = gym.make('CartPole-v0')
agent = qneuralagent(env,done_reward=-1, epsilon=0.01, epsilon_decay=1)


env.monitor.start('/tmp/cartpole-experiment-1', force=True)
for i_episode in range(2000):
    observation = env.reset()
    for t in range(100):
        #env.render()
        action = agent.getAction(observation)
        newObservation, reward, done, info = env.step(action)
        agent.update(observation, action, reward, newObservation, done)
        observation = newObservation
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.monitor.close()
gym.upload('/tmp/cartpole-experiment-1', api_key=api_key.api_key)
