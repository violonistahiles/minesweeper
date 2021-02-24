import numpy as np


class Minesweeper(object):
    '''
    Class of minesweeper game
    Input parameters:
      - mines_count - number of mines on playfield
      - playfield_h - height of playfield side
      - playfield_w - width of playfield side
      - rewards - rewards for each variant of step
    '''

    def __init__(self, mines_count, playfield_h, playfield_w, rewards):
        self.mines_count = mines_count
        self.playfield_h = playfield_h
        self.playfield_w = playfield_w
        # Actions for locate all points around current point (clockwise)
        self.actions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        self.lose_reward = rewards['lose_reward']
        self.win_reward = rewards['win_reward']
        self.yolo_reward = rewards['yolo_reward']
        self.rep_point_reward = rewards['rep_point_reward']
        self.open_point_reward = rewards['open_point_reward']
        self.free_point = set()
        self.history = []

    # Create fake play field
    def _create_playground(self):
        self.playground = np.zeros((self.playfield_h + 2, self.playfield_w + 2))
        return self.playground

    # Create filed for mines placement
    def _create_minesfield(self):
        self.minesfield = np.full((self.playfield_h, self.playfield_w), 9)
        return self.minesfield

    # Create mines in random places except first_step_coord
    def _place_mines(self, first_step_coord):

        first_point_surround = []
        for act in self.actions:
            x1 = first_step_coord[0] + act[0]
            y1 = first_step_coord[1] + act[1]
            first_point_surround.append((x1, y1))

        self.mines_coord = []
        while len(self.mines_coord) < self.mines_count:
            x = np.random.randint(0, self.playfield_h)
            y = np.random.randint(0, self.playfield_w)

            if ((x, y) != first_step_coord) and ((x, y) not in self.mines_coord) and (
                    (x, y) not in first_point_surround):
                self.mines_coord.append((x, y))

        return first_step_coord

    # Create stealth play field with mines around count
    def _mines_number(self):
        mines_round = []
        for mine in self.mines_coord:
            mines_round.append((mine[0] + 1, mine[1] + 1))

        field = np.zeros_like(self.playground)

        for i in range(1, self.playfield_h + 1):
            for j in range(1, self.playfield_w + 1):
                if (i, j) in mines_round:
                    continue

                field_round = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                               (i, j - 1), (i, j), (i, j + 1),
                               (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
                mines_count = 0
                for point in field_round:
                    mines_count += self.playground[point]
                field[i, j] = mines_count

        return field[1:-1, 1:-1]

    # Create fake play field where stored info about point surrounding
    def _create_playfield(self):
        for mine in self.mines_coord:
            x = mine[0] + 1
            y = mine[1] + 1
            self.playground[(x, y)] = 1

        self.fake_playfield = self._mines_number()

    # Check if step has reason (step in near field of open points)
    def _yolo_move(self, action):
        # action - (x,y) coordinate of current step

        for act in self.actions:
            x1 = action[0] + act[0]
            y1 = action[1] + act[1]
            if (x1, y1) in self.free_point:
                return False

        return True

    # Open all connected zeros in play field (for game acceleration)
    def open_zeros(self):

        done = True
        scale_reward = 0
        while done:
            k = 0
            for point in self.free_point.copy():
                if self.minesfield[point] == 0:

                    for act in self.actions:
                        x = point[0] + act[0]
                        y = point[1] + act[1]
                        if x in range(self.playfield_h) and y in range(self.playfield_w):
                            if self.minesfield[(x, y)] == 9:
                                k = 1
                                scale_reward += 1
                                self.free_point.add((x, y))
                                self.minesfield[(x, y)] = self.fake_playfield[(x, y)]
            if k == 0:
                return scale_reward

    # Initialize field for game
    def initialize_game(self, first_step_coord):

        self._create_playground()
        self._place_mines(first_step_coord)
        self.free_point.add(first_step_coord)

        self._create_playfield()
        self._create_minesfield()
        self.minesfield[first_step_coord] = self.fake_playfield[first_step_coord]
        _ = self.open_zeros()
        self.history.append(self.minesfield.copy())

    # Open mines field point or die
    def step(self, action):
        # action - (x,y) coordinate of current step

        x = action[0]
        y = action[1]

        # Check if current step coordinates in mines_coord
        if (x, y) in self.mines_coord:
            self.free_point.add((x, y))
            self.minesfield[(x, y)] = 11
            self.mines_coord = set(self.mines_coord) - set([(x, y)])
            for mine_coord in self.mines_coord:
                self.minesfield[(mine_coord[0], mine_coord[1])] = 10
            reward = self.lose_reward
            done = True

        # Check if current step coordinates is already done
        elif (x, y) in self.free_point:
            reward = self.rep_point_reward
            done = False

        else:
            # Add point in set of free points
            self.free_point.add((x, y))
            # Check if current step coordinates is yolo move
            if self._yolo_move((x, y)):
                self.minesfield[(x, y)] = self.fake_playfield[(x, y)]
                reward = self.yolo_reward
                done = False
            else:
                # Check if current step coordinates is last free point (win)
                if len(self.free_point) == int(self.playfield_h * self.playfield_w) - self.mines_count:
                    self.minesfield[(x, y)] = self.fake_playfield[(x, y)]
                    reward = self.win_reward
                    done = True
                else:
                    # Check if current step coordinates is not last free point
                    self.minesfield[(x, y)] = self.fake_playfield[(x, y)]
                    reward = self.open_point_reward
                    done = False

        _ = self.open_zeros()
        self.history.append(self.minesfield.copy())

        return self.minesfield, reward, done
