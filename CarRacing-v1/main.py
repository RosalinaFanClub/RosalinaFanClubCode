from dqn import Agent
import sys
import numpy as np
import gym
import tensorflow as tf

def rgb2gray(rgb):
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    return gray

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
        obs = rgb2gray(obs)
        
        frames = np.array([obs])
        frames_ = np.array([obs])

        for j in range(3):
            frames = np.append(frames, obs).reshape(j+2,96,96)
            frames_ = np.append(frames_, obs).reshape(j+2,96,96)

        for j in range(1000):
            action = agent.select_action(frames[-4:])
            obs_, reward, done, info = env.step(action)
            obs_ = rgb2gray(obs_)

            frames = np.delete(frames, slice(0,9216)).reshape(3,96,96)
            frames_ = np.delete(frames_, slice(0,9216)).reshape(3,96,96)

            frames = np.append(frames, obs).reshape(4,96,96)
            frames_ = np.append(frames_, obs_).reshape(4,96,96)
            
            agent.store(frames[-4:], action, reward, frames_[-4:], done)
            agent.learn()
            agent.learn_counter += 1
                
            if render:
                env.render()

            score += reward
            obs = obs_
            if j == 100:
                states, actions, rewards, states_, dones = agent.memory.sample_buffer()
                print(states.shape)
                print(states_.shape)
                print(action.shape)
            if score < 0:
                break
            if j % 50 == 0:
                print(score)

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: {}, score  {:.2f}, average score: {:.2f}'.format(i, score, avg_score))
    
    for i in range(10):
        done = False
        score = 0
        test_scores = []
        obs = env.reset()
        obs = rgb2gray(obs)
        
        frames = np.array([obs])
        frames_ = np.array([obs])
        
        for j in range(3):
            frames = np.append(frames, obs).reshape(j+2,96,96)
            frames_ = np.append(frames_, obs).reshape(j+2,96,96)

        for j in range(1000):
            action = agent.select_action(frames[-4:])
            obs_, reward, done, info = env.step(action)
            obs_ = rgb2gray(obs_)

            frames = np.delete(frames, slice(0,9216)).reshape(3,96,96)
            frames_ = np.delete(frames_, slice(0,9216)).reshape(3,96,96)

            frames = np.append(frames, obs).reshape(4,96,96)
            frames_ = np.append(frames_, obs_).reshape(4,96,96)
                
            env.render()

            score += reward
            obs = obs_

        test_scores.append(score)
        print('episode: {}, score  {:.2f}'.format(i, score))
    print('average score: {:.2f}'.format(np.mean(test_scores)))
