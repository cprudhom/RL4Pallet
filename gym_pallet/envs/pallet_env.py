"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class PalletEnv(gym.Env):
    """
    Description:
        Un ensemble de colis sont à placer dans un carton.
        Tous les colis sont identiques: un rectangle de dimension (l x L x h): 2 x 4 x 1.
        Un colis a un cube référence, permettant de préciser son orientation.
        Ce colis dispose de deux orientations possibles:
        - 1 : (l x L x h): 2 x 4 x 1, le cube référence est en (0,0,0),
        - 2: (l x L x h): 4 x 2 x 1, le cube référence est en (0,0,0).
        Le carton est de dimension (l x L x h): 6 x 8 x 4.
        L'objectif est de placer les 24 colis dans le carton.

    Observation:
        Type: Box(4)
        Num	Observation                             Min                             Max
        0	Carton vue de dessus                    [[.0,..,.0]...[.0,..,.0]]       [[1.,..,1.]...[1.,..,1.]]
        1   Nombre de colis restants à placer       0                               23
        2	Type de colis                           1                               1

        Note:
            La valeur d'une cellule du carton vue de dessus indique le point le plus haut sur lequel peut reposer un colis.
            Chaque colis dispose d'un cube référence, permettant de déterminer ses dimensions relatives en fonction de l'orientation.

    Actions:
        Type: Box(2)
        Num	Action
        0	k où k = y * 8 + x : position du cube référence du colis dans le carton (vue du dessus).
        1   Orientation du colis, 0 ou 1

    Récompense:
        La récompense est de 1 à chaque étape et de 100 une fois le dernier colis placé.

    Etat de départ:
        Le carton vue de dessus est valorisé à [[0.0..0.0]..[0.0..0.0]]

    Conditionns d'arrêt:
        Un colis n'est pas strictement inclus dans le carton
        Lorsque tous les colis sont placés dans le carton, le problème est considéré comme résolu.

    """

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 20
    }

    def __init__(self):
        self.palette = np.zeros(shape=(6, 8))
        self.left = 24
        self.dimensions = {0: (2, 4, 1), 1: (4, 2, 1)}
        self.slice = 1 / 4
        self.action_space = spaces.Dict(
            {'pos': spaces.Discrete(6*8),
             'ori': spaces.Discrete(2)}
        )
        self.observation_space = spaces.Dict(
            {"fill": spaces.Box(low=0., high=1., shape=(8, 6)),
             "left": spaces.Discrete(24),
             "type": spaces.Discrete(1)})
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%s (%s) invalid" % (action, type(action))
        print(action)
        k = action['pos']
        y = k // 8
        x = k % 8
        dx, dy, dz = self.dimensions[action['ori']]
        done = False
        # check bounds on (x,y)
        done = x < 0 or y < 0 or x + dx > 6 or y + dy > 8
        # check on z
        area = []
        max = np.float64(0.)
        for i in range(x, x + dx):
            if i >= 6:
                continue
            for j in range(y, y + dy):
                if j >= 8:
                    continue
                if self.palette[i,j] >= 1.:
                    done = True
                    pass
                else:
                    self.palette[i, j] += np.multiply(self.slice, dz)
                    if max < self.palette[i, j]:
                        max = self.palette[i, j]
                    area.append((i,j))

        for (i, j) in area:
            self.palette[i, j] = max
        self.left -= 1

        # do something
        self.state = {"fill": self.palette,
                      "left": self.left,
                      "type": 0}
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        self.palette = np.zeros(shape=(6, 8))
        self.left = 24
        self.state = {"fill": self.palette,
                      "left": self.left,
                      "type": 0}
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        blocksize = 30
        cx = screen_width / 2
        cy = screen_height / 2
        rx = cx - 3 * blocksize
        ry = cy - 4 * blocksize

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.grid = np.ndarray(shape=(6, 8), dtype=gym.envs.classic_control.rendering.FilledPolygon)
            for (x, y), value in np.ndenumerate(self.palette):
                cell = rendering.FilledPolygon(
                    [(rx + x * blocksize, ry + y * blocksize),
                     (rx + (x + 1) * blocksize, ry + y * blocksize),
                     (rx + (x + 1) * blocksize, ry + (y + 1) * blocksize),
                     (rx + x * blocksize, ry + (y + 1) * blocksize),
                     (rx + x * blocksize, ry + y * blocksize)])
                cell.set_color(1, .5, .5)
                self.viewer.add_geom(cell)
                self.grid[x, y] = cell

            # draw matrix
            self.gridX = []
            self.gridY = []
            for x in range(7):
                line = rendering.Line(start=(rx + (x * blocksize), ry), end=(rx + (x * blocksize), ry + 8 * blocksize))
                self.viewer.add_geom(line)
                self.gridX.append(line)
            for y in range(9):
                line = rendering.Line(start=(rx, ry + y * blocksize), end=(rx + 6 * blocksize, ry + y * blocksize))
                self.viewer.add_geom(line)
                self.gridY.append(line)

        if self.state is None: return None

        # update color wrt to fulfilment
        for (x, y), value in np.ndenumerate(self.palette):
            cell = self.grid[x, y]
            #col = rd.uniform(0., 1.)
            col = self.palette[x,y]
            cell.set_color(col, col, col)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
