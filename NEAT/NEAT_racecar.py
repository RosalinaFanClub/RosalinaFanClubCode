import itertools as it
from tabnanny import verbose
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


def lidar(frame, env):
    '''
    Get distances from car to track (left, right, and front)
    '''
    frame = frame[:84]
    left, right, front = 0, 0, 0
    for x in range(44,0,-1):
        if not frame[66][x]:
            left += 1
        else: break
    for x in range(51,96):
        if not frame[66][x]:
            right += 1
        else: break
    for y in range(67,-1,-1):
        if not frame[y][47]:
            front += 1
        else: break

    f_left, f_right = 0, 0
    for y in range(45,-1,-1):
        if not frame[y+19][y]:
            f_left += 1
        else: break

    for y, y2 in enumerate(range(65,96-66,-1)):
        if not frame[y2][y+50]:
            f_right += 1
        else: break
    
    l_front, r_front = 0, 0
    for y in range(66,-1,-2):
        if not frame[y][y//2+11]:
            l_front += 1
        else: break
    for y, y2 in enumerate(range(66,-1,-2)):
        if not frame[y2][y+51]:
            r_front += 1
        else: break
    speed = np.sqrt(np.square(env.car.hull.linearVelocity[0]) + np.square(env.car.hull.linearVelocity[1]))
    return tuple([speed, left, right, front, f_left, f_right, l_front, r_front])


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

    # action = [1,1,0]
    actions = [[0,1,0] for i in range(40)]
    for i in range(100):
        actions.append([0,0,0])
    obs = rgb2gray(obs)
    

    # while not collision(obs, ep):
    for action in actions:
        ep += 1
        obs, r, _, _ = env.step(action)
        obs = rgb2gray(obs)
        env.render()
        # if ep > 30:
        #     print_obs(obs, 0,96,0,84)
        #     print(lidar(obs, env))
        #     return

        print(ep)
def flip(env):
    '''
    Put in place to test mechanics of the game
    '''
    actions = []
    for i in range(250):
        if i < 20:
            actions.append([0,1,0])
        elif i < 40:
            actions.append([1,1,0])
        elif i < 70:
            actions.append([0,0,.2])
        elif i < 82:
            actions.append([0,1,0])
        elif i < 120:
            actions.append([-1,.9,.1])
        elif i < 143:
            actions.append([-1,.5,.1])
        elif i < 170:
            actions.append([0,0,.2])
        elif i < 250:
            actions.append([0,0,0])

    for action in actions:
        obs, _, _, _ = env.step(action)
        obs = rgb2gray(obs)
        env.render()


def eval_genomes(genomes, config):
    '''
    Evaluates the genomes based on the config file settings.
    genomes -> are the set genomes for a given population to be tests
    config -> the processed config file
    '''

    print('Evaluating genomes...')
    env = gym.make('CarRacing-v1', verbose=False)
    
    scores = []
    # create action space
    # turn = [-1, -.5, -.1, 0, .1, .5, 1]
    # gas = [1, .5]
    # brake = [0, .1, .2]
    # actions = np.array([action for action in it.product(turn, gas, brake)])
    actions = [[-1, 0, 0], [-.5, 0, 0], [-.1, 0, 0], [.1, 0, 0], [.5, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, .2]]
    
    # loop through genomes
    for id, g in genomes:
        score = 0
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        
        obs = env.reset()
        obs = rgb2gray(obs)

        for step in range(1000):
            # determine action based on observation
            action_idx = np.argmax(net.activate(lidar(obs, env)))

            # make action
            obs_, reward, _, _ = env.step(actions[action_idx])
            obs = rgb2gray(obs_)
            if RENDER:
                env.render()
            if VERBOSE:
                print(actions[action_idx])
                print('step {}: {:.2f}'.format(step, lidar(obs, env)[0]))

            # calculate fitness
            # g.fitness += reward                       # mimic score

            # g.fitness += sum(lidar(obs, env)) / 200   # reward keeping distance from sides
            speed = lidar(obs, env)[0]
            g.fitness += speed / 100      # reward speed
            # g.fitness += (actions[action_idx][1] - actions[action_idx][2]) * (score/900)
            
            score += reward
            
            if step > 50 and speed == 0:
                g.fitness -= 500
                print('Genome -> {}\tFitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))
                break
            if collision(obs, step):
                g.fitness -= 500
                print('Genome -> {}\tFitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))
                break

        scores.append(score)


def run(config):
    '''
    Actually runs a generation and calls the eval_genomes function to evaluate it
    config -> the desired config file
    '''
    print('Generating genetic matter...')
    
    if CHECKPOINT:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-2')
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

    env = gym.make('CarRacing-v1') 
    step = 0
    score = 0
    turn = [-1, -.5, -.1, 0, .1, .5, 1]
    gas = [.5]
    brake = [0, .1, .2]
    actions = np.array([action for action in it.product(turn, gas, brake)])

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    obs = env.reset()
    obs = rgb2gray(obs)

    for t in range(1000):
        # determine action based on observation
        action_idx = np.argmax(net.activate(lidar(obs, env)))

        # make action
        obs_, reward, _, _ = env.step(actions[action_idx])
        obs = rgb2gray(obs_)
        env.render()
        
        score += reward
        if collision(obs, step):
            print('Score -> {:.2f}'.format(score))
            break
        step += 1
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
        if(arg == '-v'):
            VERBOSE = True
        if(arg == '-t'):
            scores = []
            for _ in range(5):
                scores.append(test_agent(config))
            print('Average Score -> {:.2f}'.format(np.mean(scores)))
            exit(0)
        if(arg == '-d'):
            test()
            exit(0)

    run(config)