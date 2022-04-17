from gym.envs.box2d.car_dynamics import Car
import tensorflow as tf
import itertools as it
import numpy as np
import neat
import time
import gym
import os

RENDER = True

def rgb2gray(rgb):
    '''
    Process image from color to grayscale
    rgb -> color observation to be converted
    '''
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    gray = gray / 255
    return gray.round().astype(int)

def collision(obs, frame):
    '''
    Check to see if car has gone off the track
    obs -> grayscale observation of shape (96,84) 
    frame -> current frame being checked
    '''
    if frame < 12: 
        return False
    
    car = 0
    for y in range(66, 77): 
        for x in range(45,51):
            car += round(obs[y][x])

    return False if car <= 0 else True

def test():
    '''
    Put in place to test mechanics of the game
    '''
    print('Start Game!')
    env = gym.make('CarRacing-v1', )
    ep = 0
    obs = env.reset()

    action = [0,1,0]

    obs = rgb2gray(obs)

    while not collision(obs, ep):
        ep += 1
        print(collision(obs, ep))
        obs, _, _, _ = env.step(action)
        obs = rgb2gray(obs)
        env.render()

        print(ep)


def eval_genomes(genomes, config):
    '''
    Evaluates the genomes based on the config file settings.
    genomes -> are the set genomes for a given population to be tests
    config -> the processed config file
    '''
    print('Evaluating genomes...')
    env = gym.make('CarRacing-v1')
    '''
    Defines an action space of more variety and variance, size=12
    IMPORTANT need to update config file num_outputs if action space is changed
    left_right = [-1, 0, 1]
    acceleration = [1, 0]
    brake = [0.2, 0]
    actions = np.array([action for action in it.product(left_right, acceleration, brake)])
    '''
    # smaller action space (potentially more realistic actions to be taken)
    actions = [[-1,1, 0],[0,1, 0],[1,1, 0],  # left, accelerate, right
               [0,1,.2], [0,0, .2],[0,0, 0]] # maintain, brake, nothing
    scores = []
    
    # loop through genomes
    for id, g in genomes:
        score = 0
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)

        obs = env.reset()
        obs = rgb2gray(obs)

        for i in range(1000):

            # determine move based on observation
            inputs = obs[:84].reshape(8064)
            action_idx = np.argmax(net.activate(inputs))

            # make action
            obs_, reward, _, _ = env.step(actions[action_idx])
            obs = rgb2gray(obs_)
            if RENDER:
                env.render()
            
            # NEED TO IMPROVE FITNESS HEURISTIC
            g.fitness += actions[action_idx][1] # reward for speed
            g.fitness -= actions[action_idx][2] # punish for brake
            # g.fitness -= .1                   # punish for time
            if actions[action_idx][1] == 0:          
                g.fitness -= 10                 # punish for stop

            score += reward                     # update separate score

            # test for collisions
            if collision(obs, i):
                g.fitness -= 100                # punish for crashing
                print("Genome -> {} Fitness -> {:.2f}\tScore -> {:.2f}".format(id, g.fitness, score))
                break

        scores.append(score)


def run(config_file):
    '''
    Actually runs a generation and calls the eval_genomes function to evaluate it
    config -> the desired config file
    '''
    print('Generating genetic matter...')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)

    # reports population statistics
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 100)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    print('Starting up!')
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)

    # test()
