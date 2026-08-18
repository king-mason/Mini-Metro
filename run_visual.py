from agent import Agent
from visualizer import Visualizer
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)
WIDTH, HEIGHT = config["resolution"]
NUM_EPISODES = 500
VIZUALIZE_FREQ = 1

vis = Visualizer(WIDTH, HEIGHT)
agent = Agent(num_stations=20, replays_per_step=1)
agent.load("checkpoint_v3_best")

agent.epsilon = 0

for ep in range(NUM_EPISODES):
    vis.next_episode()
    if ep % VIZUALIZE_FREQ == 0 or ep == NUM_EPISODES - 1:
        reward = agent.train_episode(visualizer=vis)
        vis.hold()
    else:
        vis.events()
        reward = agent.train_episode()
    print(f"ep {ep + 1:4d} | reward {reward:7.1f} | epsilon {agent.epsilon:.4f}")

# agent.save()
vis.hold()
vis.close()
