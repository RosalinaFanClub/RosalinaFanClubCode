from dqn import Agent
import sys
import numpy as np
import gym
import tensorflow as tf

if __name__ == '__main__':
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

        for _ in range(1000):
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