from processing import rgb2gray, collision, lidar
import itertools as it
import numpy as np
import pickle
import neat
import gym
import sys
import os

CHECKPOINT = False
RENDER = False
VERBOSE = False
TEST = False


def eval_genomes(genomes, config):
    '''
    Evaluates the genomes based on the config file settings.
    genomes -> are the set genomes for a given population to be tests
    config -> the processed config file
    '''

    print('Evaluating genomes...')
    env = gym.make('CarRacing-v1', verbose=VERBOSE)
    
    # create action space
    actions = [[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, .2]]
    
    # loop through genomes
    for id, g in genomes:
        score = 0
        g.fitness = 0
        net = neat.nn.RecurrentNetwork.create(g, config)
        
        obs = env.reset()
        obs = rgb2gray(obs)
        level = 0
        for step in range(1000):

            # determine action based on observation
            inputs = lidar(obs, env)
            speed = inputs[0]
            action = actions[np.argmax(net.activate(inputs))]

            # make action
            obs_, reward, _, _ = env.step(action)
            obs = rgb2gray(obs_)
            if RENDER:
                env.render()
            if VERBOSE:
                print(action)
                print('step {} -> speed {:.2f}'.format(step, inputs[0]))

            # calculate fitness
            g.fitness += rewards
            if score // 50 != level:             # ever increasing checkpoint rewards
                g.fitness += score // 50 * 25
                level = score // 50

            score += reward     # calculate current genomes score individually
            
            if step > 50 and speed < 5:     # terminate if insufficient speed
                g.fitness -= 500
                print('Genome -> {}\tFitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))
                break
            if collision(obs, step):        # terminate if crashed
                g.fitness -= 500
                print('Genome -> {}\tFitness -> {:.2f}\tScore -> {:.2f}   Speed -> {:.2f}'.format(id, g.fitness, score, speed))
                break
            if step == 999:                 # reward surviving for the duration of the episode
                g.fitness += 100
                print('TIMED OUT!\tGenome -> {}\tFitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))
                
        scores.append(score)


def run(config, check=0):
    '''
    Actually runs a generation and calls the eval_genomes function to evaluate it
    config -> the desired config file
    '''
    print('Generating genetic matter...')
    
    if CHECKPOINT:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + check)
    else:
        p = neat.Population(config)

    # reports population statistics
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    best = p.run(eval_genomes, 100)
    with open('best.pickle', 'wb') as f:
        pickle.dump(best, f)

    print('\nBest genome:\n{!s}'.format(best))


def test_agent(config, file='recurrent.pickle'):
    with open(file, 'rb') as f:
        best = pickle.load(f)

    env = gym.make('CarRacing-v1') 
    step = 0
    score = 0
    actions = [[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, .2]]

    net = neat.nn.RecurrentNetwork.create(best, config)
    obs = env.reset()
    obs = rgb2gray(obs)

    for step in range(1000):
        # determine action based on observation
        inputs = lidar(obs, env)
        speed = inputs[0]
        action = actions[np.argmax(net.activate(inputs))]

        # make action
        obs_, reward, _, _ = env.step(action)
        obs = rgb2gray(obs_)
        env.render()
        
        score += reward
        if collision(obs, step):
            print('Speed -> {:.2f}'.format(speed), end=' ')
            break

    print('Score -> {:.2f}'.format(score))
    return score


if __name__ == '__main__':
    print('Starting up!')

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    for arg in sys.argv:
        if(arg == '-r'):
            RENDER = True
        if(arg == '-c'):
            CHECKPOINT = True
            idx = sys.argv.index('-c')
            run(config, check=sys.argv[idx+1])
            exit(0)
        if(arg == '-v'):
            VERBOSE = True
        if(arg == '-t'):
            idx = sys.argv.index('-t')
            file = sys.argv[idx+1]
            scores = []
            for _ in range(10):
                scores.append(test_agent(config))
            print('Average Score -> {:.2f}'.format(np.mean(scores)))
            exit(0)

    run(config)