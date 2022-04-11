from dqn import Agent
import sys
import numpy as np
import gym
import tensorflow as tf
from PIL import ImageOps

def rgb2gray(rgb):
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    gray = gray / 128. - 1.
    return np.array(3 * [gray])

if __name__ == '__main__':
    render = False
    if(len(sys.argv) > 1):
        if(sys.argv[1] == '-r'):
            render = True

    env = gym.make('CarRacing-v1', )
    agent = Agent()
    
    n_games = 500
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        print(len(obs), len(obs[0]), len(obs[0][0]))
        obs = rgb2gray(obs)
        print(len(obs), len(obs[0]), len(obs[0][0]))
        for j in range(1000):
            if j % 50 == 0: # to show progress
                print(j)
            
            action = agent.select_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.store(obs, action, reward, obs_, done)
            agent.learn()
            
            if render:
                env.render()

            score += reward
            obs = obs_

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: {}, score  {:.2f}, average score: {:.3f}'.format(i, score, avg_score))
    
    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()

        for _ in range(500):
            action = agent.select_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.store(obs, action, reward, obs_, done)

            score += reward
            obs = obs_
            env.render()

        print('episode: {}, score  {:.2f}'.format(i, score))