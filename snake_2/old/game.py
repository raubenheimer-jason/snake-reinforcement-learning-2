import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
# font = pygame.font.Font('arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Encode(Enum):
    """ positio in the input channels, almost like RGB for image... """
    SNAKE_HEAD_POS = int(0)
    SNAKE_BODY_POS = int(1)
    FOOD_POS = int(2)


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        # make sure not to place the food in the snake body
        if self.food in self.snake:
            # if the food is in the snake call method again,
            # try place food again until it is not in snake body
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 100
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    # def is_trapped(self):
    #     """ check if the snake is trapped
    #         - returns True if it will eventually eat itsself given its current head and body position
    #     """

    #     for idx, pt in enumerate(self.snake):

    #     if pt is None:
    #         pt = self.head
    #     # hits boundary
    #     if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
    #         return True
    #     # hits itself
    #     if pt in self.snake[1:]:
    #         return True

    #     return False

    # def get_flat_grid_snake(self):
    #     """ converts the pixel xy into grid position xy """

    #     x = self.w//BLOCK_SIZE
    #     y = self.h//BLOCK_SIZE

    #     # print(f"x: {x}  y: {y}")

    #     grid_snake = [[0 for _ in range(x)] for _ in range(y)]

    #     for pt in self.snake:
    #         grid_x = int(pt.x/BLOCK_SIZE)-1
    #         grid_y = int(pt.y/BLOCK_SIZE)-1
    #         # print(f"grid_x: {grid_x}  grid_y: {grid_y}")
    #         grid_snake[grid_y][grid_x] = 1

    #     # flatten grid_snake
    #     flat_grid_snake = [x for xs in grid_snake for x in xs]

    #     return flat_grid_snake

    def get_grid_state(self):
        """ Gets the state of all the "blocks" in the grid 
            - snake body position
            - food position
            - snake head position?


        """

        x = self.w//BLOCK_SIZE
        y = self.h//BLOCK_SIZE

        # print(f"x: {x}  y: {y}")

        # 2D array of grid
        # each pos has array for the 3 channels (almost like RGB...)
        # grid = [[[0, 0, 0] for _ in range(x)] for _ in range(y)]
        grid = [[0 for _ in range(x)] for _ in range(y)]

        # "encode" snake position in grid
        for idx, pt in enumerate(self.snake):
            snake_grid_x = int(pt.x/BLOCK_SIZE)-1
            snake_grid_y = int(pt.y/BLOCK_SIZE)-1

            if idx == 0:
                # this is the head, store different value?
                # grid[snake_grid_y][snake_grid_x][Encode.SNAKE_HEAD_POS] = 1
                # grid[snake_grid_y][snake_grid_x][0] = 1
                grid[snake_grid_y][snake_grid_x] = 1
            else:
                # grid[snake_grid_y][snake_grid_x][Encode.SNAKE_BODY_POS] = 1
                # grid[snake_grid_y][snake_grid_x][1] = 1
                grid[snake_grid_y][snake_grid_x] = 2

        # "encode" food in grid
        food_grid_x = int(self.food.x/BLOCK_SIZE)-1
        food_grid_y = int(self.food.y/BLOCK_SIZE)-1
        # grid[food_grid_y][food_grid_x][Encode.FOOD_POS] = 1
        # grid[food_grid_y][food_grid_x][2] = 1
        grid[food_grid_y][food_grid_x] = 3

        # # flatten grid_snake
        # flat_grid_snake = [x for xs in grid for x in xs]

        # return flat_grid_snake
        return grid

    def _update_ui(self):
        self.display.fill(BLACK)

        # print(self.snake)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, righ, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # clockwise (right turn)
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # counter clockwise (left turn)

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


# if __name__ == '__main__':
#     game = SnakeGame()

#     # game loop
#     while True:
#         game_over, score = game.play_step()

#         if game_over == True:
#             break

#     print('Final Score', score)

#     pygame.quit()
