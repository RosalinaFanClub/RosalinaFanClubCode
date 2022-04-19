import itertools as it
import numpy as np
import neat
import gym
import sys
import os

CHECKPOINT = False
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

    action = [0,1,0]

    obs = rgb2gray(obs)
    print_obs(obs, 0,96,0,84)
    print_obs(obs, 45,51,66,77)

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

    scores = []
    for id, g in genomes:
        score = 0
        speed_sum = 0
        g.fitness = 0
        
        net = neat.nn.FeedForwardNetwork.create(g, config)

        obs = env.reset()
        obs = rgb2gray(obs)

        for t in range(1000):
            # determine action based on observation
            output = net.activate(lidar(obs))

            # process action
            turn_output = output[0]
            gas_output = .5
            brake_output = .1

            action = (turn_output, gas_output, brake_output)

            # make action
            obs_, reward, _, _ = env.step(action)
            obs = rgb2gray(obs_)
            if RENDER:
                env.render()
                # print(action)


            speed_sum += (gas_output - brake_output)
            
            g.fitness += (gas_output - brake_output)
            if gas_output == 0:
                g.fitness -= 10
            # if np.floor(speed_sum * t) % 100 == 0:
            #     g.fitness += 100
            score += reward

            if collision(obs, t):
                g.fitness -= 100
                # print('step -> {} speed sum -> {:.2f}'.format(t, speed_sum))
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
        p = neat.Checkpointer.restore_checkpoint('continuous-checkpoint')
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 100)

    print('\nBest genome:\n{!s}'.format(winner))


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
        if(arg == '-d'):
            test()
            exit(0)

    run(config)