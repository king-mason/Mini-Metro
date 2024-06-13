from game import MetroGame
import game_detection as gd
from station import Station
from point import Point

import os
import time
import pickle
import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque


def flatten_state(state_observation):
    trains = state_observation[0]
    train_cars = state_observation[1]
    tunnels = state_observation[2]
    waiting = state_observation[3]
    lines = state_observation[4]
    stations = state_observation[5]

    station_locations = []
    for station in stations:
        station_locations.append(station.shape_index)
        station_locations.append(station.position.x)
        station_locations.append(station.position.y)
    station_tensor = torch.FloatTensor(station_locations)

    # Aggregate the station info
    pooled_station_embeddings = torch.mean(station_tensor, dim=0).unsqueeze(0)

    line_embeddings = []
    for line in lines:
        line_positions = torch.FloatTensor([station.position for station in line]).flatten()
        line_embeddings.append(torch.mean(line_positions, dim=0))
    
    if line_embeddings:
        pooled_line_embeddings = torch.mean(torch.stack(line_embeddings), dim=0).unsqueeze(0)
    else:
        pooled_line_embeddings = torch.tensor([0]).unsqueeze(0)
    
    return [trains, train_cars, tunnels, waiting, *pooled_line_embeddings.flatten(), *pooled_station_embeddings.flatten()]

def update_action_mappings(stations):
    num_stations = len(stations)
    action_to_station = [(i, j) for i in range(num_stations + 2) for j in range(num_stations + 2) if i != j]
    station_to_action = {pair: idx for idx, pair in enumerate(action_to_station)}
    return action_to_station, station_to_action

def get_random_action(stations):
    n = len(stations) + 2
    action_idx = random.randint(0, (n * (n-1)) - 1)
    action_to_station, _ = update_action_mappings(stations)
    return action_to_station[action_idx]

def action_to_stations(action, stations):
    station_1_idx, station_2_idx = action
    if station_1_idx == 0:
        station_1 = Point(490, 935)
    elif station_1_idx == 1:
        station_1 = Point(420, 935)
    else:
        station_1 = stations[station_1_idx - 2]
    if station_2_idx == 0:
        station_2 = Point(490, 935)
    if station_2_idx == 1:
        station_2 = Point(420, 935)
    else:
        station_2 = stations[station_2_idx - 2]
    return station_1, station_2

def line_extension(station_1, station_2, lines):
    if type(station_1) is not Station or type(station_2) is not Station:
        return False, 0, None, None
    
    for i, line in enumerate(lines):
        if len(line) < 2:
            continue
        if line[0] == station_1 or line[-1] == station_1:
            return True, i, station_1, station_2
        if line[0] == station_2 or line[-1] == station_2:
            return True, i, station_2, station_1

    return False, 0, station_1, station_2
        

def check_new_week():
    new_train_loc, new_train_size = gd.locate_on_screen('screenshots/new_locomotive.png')
    if new_train_loc:
        gd.move_to_center(new_train_loc, new_train_size)
        pyautogui.click(button='left')
        pyautogui.moveTo(750, 935)
        time.sleep(1)
        options = [gd.locate_on_screen('screenshots/new_line.png'), gd.locate_on_screen('screenshots/new_tunnels.png'),
                    gd.locate_on_screen('screenshots/new_carriage.png')]
        
        for i in range(10):
            pick = random.randint(0, 2)
            loc, size = options[pick]
            if loc:
                gd.move_to_center(loc, size)
                pyautogui.click(button='left')
                time.sleep(1)
                break
        else:
            return False
    else:
        return False
    return True


# USE LATER
class ReplayBuffer:
    def __init__(self, data_names, buffer_size, mini_batch_size):
        self.data_keys = data_names
        self.data_dict = {}
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.reset()

    def reset(self):
        # Create a deque for each data type with set max length
        for name in self.data_keys:
            self.data_dict[name] = deque(maxlen=self.buffer_size)

    def buffer_full(self):
        return len(self.data_dict[self.data_keys[0]]) == self.buffer_size

    def data_log(self, data_name, data):
        # split tensor along batch into a list of individual datapoints
        data = data.cpu().split(1)
        # Extend the deque for data type, deque will handle popping old data to maintain buffer size
        self.data_dict[data_name].extend(data)

    def __iter__(self):
        batch_size = len(self.data_dict[self.data_keys[0]])
        batch_size = batch_size - batch_size % self.mini_batch_size

        ids = np.random.permutation(batch_size)
        ids = np.split(ids, batch_size // self.mini_batch_size)
        for i in range(len(ids)):
            batch_dict = {}
            for name in self.data_keys:
                c = [self.data_dict[name][j] for j in ids[i]]
                batch_dict[name] = torch.cat(c)
            batch_dict["batch_size"] = len(ids[i])
            yield batch_dict

    def __len__(self):
        return len(self.data_dict[self.data_keys[0]])

# MLP Q Network
class QNet(nn.Module):
    def __init__(self, input_num=4, action_num=2, hidden_size=64):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

class Agent:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []  # For experience replay
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.5  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Initialize replay buffer
        buffer_size = 3000
        mini_batch_size = 32
        data_names = data_names = ["states", "next_states", "actions", "rewards", "masks"]
        self.replay_buffer = ReplayBuffer(data_names, buffer_size, mini_batch_size)

        # Initialize q network
        stations = 3
        # observation_size = (stations * 3) + 5
        # self.action_size = stations ** 2
        self.observation_size = 12
        self.action_size = 1
        action_size_2 = 2  # for choosing utilities
        self.q_net = QNet(self.observation_size, self.action_size)
        # self.q_net_2 = QNet(observation_size, action_size_2)
        self.target_net = QNet(self.observation_size, self.action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        # Initialize the environment
        self.env = MetroGame()
        self.load("checkpoint")

    def valid_actions(self, state):
        actions = np.array([1, 1, 1, 0, 0])
        lines_unused = [line for line in state[1] if not line]
        if state[2] > lines_unused:
            # able to add a train
            actions[3] = 1
        if state[3] > 0:
            # able to add a train car
            actions[4] = 1
        return np.where(actions == 1)[0]
    
    def select_best_action(self, state, stations):
        q_values = []

        max_num_stations = 100
        embedding_dim = 3  # Size of the embedding vectors

        station_embedding_layer = nn.Embedding(max_num_stations, embedding_dim)

        # Map actions to station pairs
        action_to_station, _ = update_action_mappings(stations)
        num_actions = len(action_to_station)

        # Convert state to tensor
        state_tensor = torch.FloatTensor(flatten_state(state)).unsqueeze(0)
        
        # Evaluate Q-values for all possible actions
        for action_idx in range(num_actions):
            station_pair = action_to_station[action_idx]
            station_pair_tensor = torch.LongTensor(station_pair).unsqueeze(0)

            # Get embeddings for the stations
            station1_embed = station_embedding_layer(station_pair_tensor[:, 0])
            station2_embed = station_embedding_layer(station_pair_tensor[:, 1])
            
            # Concatenate state and action embeddings
            state_action_tensor = torch.cat((state_tensor, station1_embed, station2_embed), dim=1)
            
            # Get Q-value from the Q-network
            if len(state_action_tensor.flatten()) != self.observation_size:
                print(f'Invalid input space: size {len(state_action_tensor.flatten())}')
                continue
            q_value = self.q_net(state_action_tensor)
            q_values.append(q_value.item())
        if q_values:
            action_idx = np.argmax(q_values)
            return action_to_station[action_idx]
        else:
            print('No Q Values')
            return get_random_action(stations)

    def replay(self, batch_size):
        
        if len(self.memory) < batch_size:
            return

        minibatch = self.sample_from_memory(batch_size)
        
        max_num_stations = 100
        embedding_dim = 3
        station_embedding_layer = nn.Embedding(max_num_stations, embedding_dim)
        action_to_station, station_to_action = update_action_mappings(self.env.stations)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(flatten_state(state)).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(flatten_state(next_state)).unsqueeze(0)

            print(action)
            station_pair = action
            station_pair_tensor = torch.LongTensor(station_pair).unsqueeze(0)

            station1_embed = station_embedding_layer(station_pair_tensor[:, 0])
            station2_embed = station_embedding_layer(station_pair_tensor[:, 1])

            state_action_tensor = torch.cat((state_tensor, station1_embed, station2_embed), dim=1)
            next_state_action_tensor = torch.cat((next_state_tensor, station1_embed, station2_embed), dim=1)

            target = reward
            if not done:
                next_q_values = self.target_net(next_state_action_tensor)
                target = reward + self.gamma * torch.max(next_q_values).item()

            action_idx = station_to_action[action]
            current_q_values = self.q_net(state_action_tensor)
            target_q_values = current_q_values.clone()
            target_q_values[0][action_idx] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def sample_from_memory(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def save(self, name):
        if not os.path.exists(name):
            os.makedirs(name)
        torch.save(self.q_net.state_dict(), os.path.join(name, 'q_net.pth'))
        torch.save(self.target_net.state_dict(), os.path.join(name, 'target_net.pth'))
        with open(os.path.join(name, 'memory.pkl'), 'wb') as f:
            pickle.dump(self.memory, f)
        hyperparameters = {
            'epsilon': self.epsilon,
            # 'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        with open(os.path.join(name, 'hyperparameters.pkl'), 'wb') as f:
            pickle.dump(hyperparameters, f)
        print("Model, replay buffer, and hyperparameters saved.")

    def load(self, name):
        q_net_path = os.path.join(name, 'q_net.pth')
        target_net_path = os.path.join(name, 'target_net.pth')
        memory_path = os.path.join(name, 'memory.pkl')
        hyperparameters_path = os.path.join(name, 'hyperparameters.pkl')
        if os.path.exists(q_net_path):
            self.q_net.load_state_dict(torch.load(q_net_path))
            self.target_net.load_state_dict(torch.load(target_net_path))
            print("Model loaded.")
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
            print("Replay buffer loaded.")
        if os.path.exists(hyperparameters_path):
            with open(hyperparameters_path, 'rb') as f:
                hyperparameters = pickle.load(f)
            self.epsilon = hyperparameters['epsilon']
            print("Hyperparameters loaded.")


    def play(self):
        # Plays the game until reaching a game over state
        env = self.env
        done = False

        states = []
        rewards = []
        actions = []
        masks = []

        env.reset()
        pyautogui.moveTo(550, 950)
        pyautogui.moveTo(1000, 950, duration=1)

        while not done:
            if gd.get_white_area() < 0.5:
                print('Game not detected')
                time.sleep(3)
                continue
            state = [env.trains, env.train_cars, env.tunnels, env.waiting_passengers, env.lines, env.stations]
            if len(state[5]) == 0:
                if check_new_week():
                    env.update()
                    state = [env.trains, env.train_cars, env.tunnels, env.waiting_passengers, env.lines, env.stations]
                else:
                    raise Exception('No stations found.')
            flat_state = torch.FloatTensor(flatten_state(state)).unsqueeze(0)
            states.append(flat_state)

            if random.random() <= self.epsilon:
                # Pick random move
                action = get_random_action(env.stations)
            else:
                # Pick best move from q network
                action = self.select_best_action(state, env.stations)
            
            station_1, station_2 = action_to_stations(action, env.stations)

            extend, line_idx, from_station, to_station = line_extension(station_1, station_2, env.lines)

            try:
                if extend:
                    env.extend_line(line_idx, from_station, to_station)
                elif (from_station is not None) and (to_station is not None):
                    env.create_line([from_station, to_station])
                elif type(station_1) is Point and type(station_2) is Station:
                    env.connect(station_1, station_2.position)
                elif type(station_1) is Station and type(station_2) is Point:
                    env.connect(station_1.position, station_2)
            except:
                print('Error making move')

            # Give the game time to respond to action
            pyautogui.moveTo(550, 950)
            pyautogui.moveTo(1000, 950, duration=3)

            check_new_week()

            reward, done = env.update()
            next_state = [env.trains, env.train_cars, env.tunnels, env.waiting_passengers, env.lines, env.stations]

            if done:
                next_state = state
                env.reset()

            self.remember(state, action, reward, next_state, done)
            

        # self.replay(batch_size=32)

        # Adjust exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.save("checkpoint")

        # self.replay_buffer.data_log("states", torch.cat(states[:-1]))
        # self.replay_buffer.data_log("next_states", torch.cat(states[1:]))
        # self.replay_buffer.data_log("rewards", torch.cat(rewards))
        # self.replay_buffer.data_log("actions", torch.cat(actions))
        # self.replay_buffer.data_log("masks", torch.cat(masks))





if __name__ == '__main__':

    # Example
    agent = Agent(100)

    # Example state
    stations = [Station("circle", Point(i, i)) for i in range(3)]  # Dummy stations
    state = [1, 2, 3, 4, [[Station("circle", Point(i, i)) for i in range(3)]], stations]  # Dummy state

    flat_state = flatten_state(state)
    print(len(flat_state))
    print(flat_state)

    # Select best action
    best_action = agent.select_best_action(state, stations)
    
    print(f"Best action: {best_action}")


    # Play the actual game
    for i in range(20):
        agent.play()
