import itertools as it
import numpy as np
import pickle
import neat
import gym
import sys
import os

CHECKPOINT = False
RENDER = False
TEST = False


def rgb2gray(rgb):
    '''
    Process image from color to grayscale
    -rgb -> color observation to be converted
    '''
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    gray = gray / 255
    return gray.round().astype(int)


def collision(obs, frame):
    '''
    Check to see if car has gone off the track
    -obs -> grayscale observation of shape (96,84) 
    -frame -> current frame being checked
    '''
    if frame < 20: 
        return False
    
    car = 0
    for y in range(66, 77): 
        for x in range(45,51):
            car += obs[y][x]

    return False if car <= 1 else True


def lidar(frame):
    '''
    Get distances from car to track (left, right, and front)
    '''
    frame = frame[:84]
    left, right, front = 0, 0, 0
    for x in range(35,46):
        if not frame[66][x]:
            left += 1
    for x in range(51,62):
        if not frame[66][x]:
            right += 1
    for y in range(0, 67):
        if not frame[y][47]:
            front += 1
    return tuple([left, right, front])


def print_obs(obs, x_min=0, x_max=96, y_min=0, y_max=96):
    '''
    Print a specific section of an observation
    '''
    obs[76][45] = 2
    obs[66][45] = 2
    obs[76][50] = 2
    obs[66][50] = 2
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            print('{:.0f} '.format(obs[y][x]), end='')
        print()
def test():
    '''
    Put in place to test mechanics of the game
    '''
    print('Start Game!')
    env = gym.make('CarRacing-v1', )
    ep = 0
    obs = env.reset()

    action = [0,0,0]

    obs = rgb2gray(obs)
    

    while not collision(obs, ep):
        ep += 1
        print(collision(obs, ep))
        obs, _, _, _ = env.step(action)
        obs = rgb2gray(obs)
        env.render()
        if ep > 30:
            print_obs(obs, 0,96,0,84)
            print(lidar(obs))
            
            return

        print(ep)


def eval_genomes(genomes, config):
    '''
    Evaluates the genomes based on the config file settings.
    genomes -> are the set genomes for a given population to be tests
    config -> the processed config file
    '''

    print('Evaluating genomes...')
    env = gym.make('CarRacing-v1')
    track = env.track
    # Defines an action space of more variety and variance, size=12
    # IMPORTANT need to update config file num_outputs if action space is changed
    # left_right = [-1, 0, 1]
    # acceleration = [1, 0]
    # brake = [0.2, 0]
    # actions = np.array([action for action in it.product(left_right, acceleration, brake)])
    
    # smaller action space (potentially more realistic actions to be taken)
    actions = [[-1,.5, .1],[0,5, .1],[1,.5, .1], [0,0,.2]]
    scores = []
    
    # loop through genomes
    for id, g in genomes:
        score = 0
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        
        obs = env.reset()
        env.track = track
        obs = rgb2gray(obs)

        for i in range(1000):
            # determine action based on observation
            action_idx = np.argmax(net.activate(lidar(obs)))

            # make action
            obs_, reward, _, _ = env.step(actions[action_idx])
            obs = rgb2gray(obs_)
            if RENDER:
                env.render()
                # print(actions[action_idx])
            

            g.fitness += sum(lidar(obs))/2

            # g.fitness += (actions[action_idx][1] - actions[action_idx][2])
            if actions[action_idx][1] == 0:
                g.fitness -= 20
            # g.fitness += reward
            score += reward

            if collision(obs, i):
                g.fitness -= 100               
                print('Genome -> {} Fitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))
                break

        scores.append(score)


def run(config):
    '''
    Actually runs a generation and calls the eval_genomes function to evaluate it
    config -> the desired config file
    '''
    print('Generating genetic matter...')
    
    if CHECKPOINT:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
    else:
        p = neat.Population(config)

    # reports population statistics
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 100)
    with open('best.pickle', 'wb') as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))


def test_agent(config, file='best.pickle'):
    with open(file, 'rb') as f:
        winner = pickle.load(f)

    score = 0
    env = gym.make('CarRacing-v1')
    actions = [[-1,.5, .1],[0,5, .1],[1,.5, .1],[0,0,.2]]
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    obs = env.reset()
    obs = rgb2gray(obs)

    for t in range(1000):
        # determine action based on observation
        action_idx = np.argmax(net.activate(lidar(obs)))

        # make action
        obs_, reward, _, _ = env.step(actions[action_idx])
        obs = rgb2gray(obs_)
        env.render()
        
        score += reward

        if collision(obs, t):
            break
    print('Score -> {:.2f}'.format(score))


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
        if(arg == '-t'):
            for _ in range(5):
                test_agent(config)
            exit(0)
        if(arg == '-c'):
            CHECKPOINT = True
        if(arg == '-d'):
            test()
            exit(0)

    run(config)