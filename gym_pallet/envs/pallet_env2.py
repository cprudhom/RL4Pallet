"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class PalletEnv2(gym.Env):
    """
    Description:
        Un ensemble de colis sont à placer dans un carton.
        Tous les colis sont identiques: un rectangle de dimension (l x L x h): 1 x 2 x 1.
        Un colis a un cube référence, permettant de préciser son orientation.
        Ce colis dispose de deux orientations possibles:
        - 1 : (l x L x h): 1 x 2 x 1, le cube référence est en (0,0,0),
        - 2: (l x L x h): 2 x 1 x 1, le cube référence est en (0,0,0).
        Le carton est de dimension (l x L x h): 3 x 4 x 4.
        L'objectif est de placer les 24 colis dans le carton.

    Observation:
        Type: Box(4)
        Num	Observation                             Min                             Max
        0	Carton vue de dessus                    [[.0,..,.0]...[.0,..,.0]]       [[4.,..,4.]...[4.,..,4.]]
        1	Type de colis                           1                               1

        Note:
            La valeur d'une cellule du carton vue de dessus indique le point le plus haut sur lequel peut reposer un colis.
            Chaque colis dispose d'un cube référence, permettant de déterminer ses dimensions relatives en fonction de l'orientation.

    Actions:
        Type: Box(2)
        Num	Action
        0	Position sur l'axe des abscisses du cube référence du colis dans le carton (vue du dessus).
        1	Position sur l'axe des ordonnées du cube référence du colis dans le carton (vue du dessus).
        2   Orientation du colis, 0 ou 1

    Récompense:
        La récompense est de 1 à chaque étape

    Etat de départ:
        Le carton vue de dessus est valorisé à [[0.0..0.0]..[0.0..0.0]]

    Conditionns d'arrêt:
        Un colis n'est pas strictement inclus dans le carton
        Lorsque le carton est plein.

    """

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 20
    }

    def __init__(self):
        self.X = 5
        self.Y = 9
        self.C = self.X * self.Y // 2
        self.box = np.zeros(shape=(self.X, self.Y), dtype=int)
        self.dimensions = {
            0: (1, 2),
            1: (2, 1)
        }
        self.action_space = spaces.Dict(
            {'pos_x': spaces.Discrete(self.X),
             'ori': spaces.Discrete(len(self.dimensions))}
        )
        self.observation_space = spaces.Dict(
            {"fill": spaces.MultiDiscrete([self.Y for _ in range(self.X)]),
             "type": spaces.Discrete(1)})
        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def prevY(self, x):
        y = self.Y - 1
        while y > -1 and self.box[x][y] == 0:
            y -= 1
        return y + 1

    def step(self, action):
        assert self.action_space.contains(action), "%s (%s) invalid" % (action, type(action))

        self.C -= 1
        x = action['pos_x']
        dx, dy = self.dimensions[action['ori']]
        if x >= 0 and x + dx <= self.X:
            if dx == 1:
                # check x and 'fill' to compute y
                y = self.prevY(x)
            else:
                # check x and x+1 and 'fill' to compute y
                y = max(self.prevY(x), self.prevY(x + 1))
        else:
            y = 0
        # check bounds on (x,y)
        done = x < 0 or x + dx > self.X or y + dy > self.Y
        # update box
        area = []
        for i in range(x, x + dx):
            if i >= self.X:
                # overload
                continue
            for j in range(y, y + dy):
                if j >= self.Y:
                    # overload
                    continue
                self.box[i][j] = 1
        self.state = {"fill": self.box,
                      "type": 0}
        last = np.sum(self.box[:,self.Y-1])
        if self.C == 0:
            done = True
            reward = 10000
        elif last >= self.X - 1:
            done = True
            reward = 1000 - self.C * 50
        elif not done:
            reward = 1.0
        else:
            reward = 0.0
        return self.state, reward, done, {}

    def reset(self):
        self.box = np.zeros(shape=(self.X, self.Y), dtype=int)
        self.state = {"fill": self.box,
                      "type": 0}
        self.C = self.X * self.Y // 2
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human'):
        screen_width = (self.Y + 1) * 100
        screen_height = (self.X + 1) * 100
        blocksize = 40
        cx = screen_width / 2
        cy = screen_height / 2
        rx = cx - self.X / 2 * blocksize
        ry = cy - self.Y / 2 * blocksize

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.grid = np.ndarray(shape=(self.X, self.Y), dtype=gym.envs.classic_control.rendering.FilledPolygon)
            for (x, y), value in np.ndenumerate(self.box):
                cell = rendering.FilledPolygon(
                    [(rx + x * blocksize, ry + y * blocksize),
                     (rx + (x + 1) * blocksize, ry + y * blocksize),
                     (rx + (x + 1) * blocksize, ry + (y + 1) * blocksize),
                     (rx + x * blocksize, ry + (y + 1) * blocksize),
                     (rx + x * blocksize, ry + y * blocksize)])
                cell.set_color(1, 1, .5)
                self.viewer.add_geom(cell)
                self.grid[x, y] = cell

            # draw matrix
            self.gridX = []
            self.gridY = []
            r, g, b = .5, .0, 1
            for x in range(self.X + 1):
                line = rendering.Line(start=(rx + (x * blocksize), ry),
                                      end=(rx + (x * blocksize), ry + self.Y * blocksize))
                line.set_color(r, g, b)
                self.viewer.add_geom(line)
                self.gridX.append(line)
            for y in range(self.Y + 1):
                line = rendering.Line(start=(rx, ry + y * blocksize), end=(rx + self.X * blocksize, ry + y * blocksize))
                line.set_color(r, g, b)
                self.viewer.add_geom(line)
                self.gridY.append(line)

        if self.state is None: return None

        # update color wrt to fulfilment
        for (x, y), value in np.ndenumerate(self.box):
            cell = self.grid[x, y]
            col = self.box[x, y]
            cell.set_color(col, col, col)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
