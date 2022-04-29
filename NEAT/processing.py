import numpy as np

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

    return False if car <= 3 else True


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