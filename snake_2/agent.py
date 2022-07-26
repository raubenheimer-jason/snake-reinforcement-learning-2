import torch
import random
import numpy as np
from collections import deque  # store memories
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
# from model import ConvNet, CnnTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_gmaes = 0
        self.epsilon = 0  # randomness
        # gamma is the discount rate (must be smaller than 1, usually .8/.9)
        self.gamma = 0.9
        # if deque exceeds MAX_MEMORY it automatically calls popleft()
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        # self.model = ConvNet()
        # self.model = Linear_QNet(768, 256, 3)
        # self.model = Linear_QNet(768, 1024, 3)
        # self.model = Linear_QNet(779, 1024, 3)  # 11 + 768 inputs
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # self.trainer = CnnTrainer(self.model, lr=LR, gamma=self.gamma)

        # grid where every block is 0 except where snake body is
        self.grid_2d_arr = [[]]

    # def get_grid(self, game):
    #     grid_state = game.get_grid_state()

    #     # # flatten grid_state
    #     flat_grid_state = [x for xs in grid_state for x in xs]
    #     return flat_grid_state

    def get_state(self, game):
        head = game.snake[0]
        # check if there is danger in all directions
        point_l = Point(head.x - 20, head.y)  # 20 is the BLOCK_SIZE
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        # information about location of snake body
        # we dont know the snake length so just add information about entire grid state
        # and whether block contains snake body or not...
        # for 640/20 x 480/20 grid size we are adding (32x24=768) states
        # state += game.get_flat_grid_snake()

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # automatically does popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # batch
        # grab 1000 samples from memory

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # only train for one game step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        # in the beginning you want more random moves
        # the better the model gets the more you want to use (and exploit) the model

        self.epsilon = 80 - self.n_gmaes
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # 2 inclusive (0, 1, or 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def get_action_cnn(self, grid_state):
        # random moves: tradeoff between exploration and exploitation
        # in the beginning you want more random moves
        # the better the model gets the more you want to use (and exploit) the model

        self.epsilon = 80 - self.n_gmaes
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # 2 inclusive (0, 1, or 2)
            final_move[move] = 1
        else:
            grid_state0 = torch.tensor(grid_state, dtype=torch.float)
            prediction = self.model(grid_state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []  # for plotting
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # runs until we quit

        # get old state
        state_old = agent.get_state(game)
        # cnn
        # get_grid_old = agent.get_grid(game)

        # get move
        final_move = agent.get_action(state_old)
        # final_move = agent.get_action_cnn(get_grid_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # cnn
        # get_grid_new = agent.get_grid(game)

        # train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)
        # agent.train_short_memory(
        #     get_grid_old, final_move, reward, get_grid_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        # agent.remember(get_grid_old, final_move, reward, get_grid_new, done)

        if done:
            # train long memory (replay memory / experience replay)
            # trains on all previous moves / games it played
            # plot result
            game.reset()
            agent.n_gmaes += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game {agent.n_gmaes}, Score {score}, Record {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_gmaes
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
