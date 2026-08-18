from agent import Agent
from visualizer import Visualizer
import yaml
import matplotlib.pyplot as plt
import numpy as np

NUM_EPISODES = 500
ROLLING_WINDOW = 50
MAP_BLOCK_SIZE = 10  # episodes per map seed before resampling
EVAL_EVERY = 25      # episodes between held-out evaluations

# Held-out maps the agent NEVER trains on. Training seeds are episode // MAP_BLOCK_SIZE
# (0..49 for 500 episodes), so this band is disjoint — it measures generalization.
TEST_SEEDS = list(range(10_000, 10_010))

agent = Agent(num_stations=20, epsilon_decay=0.994, epsilon_min=0.005,
              replays_per_step=1, k_candidates=10, num_playouts=3,
              mcts_sims=30, mcts_horizon=6)  # mcts_sims=0 disables look-ahead (greedy)
agent.load()

rewards = []
ticks = []
eval_episodes = []   # x positions (episode index) of each evaluation
eval_rewards = []    # mean held-out reward at each evaluation

# TODO:
# make user input option
map_seed = 0    
for episode in range(NUM_EPISODES):
    if episode % MAP_BLOCK_SIZE == 0:
        map_seed = np.random.randint(10000)
    reward = agent.train_episode(map_seed=map_seed)
    rewards.append(reward)
    ticks.append(agent.last_episode_ticks)
    print(f"Episode {episode + 1:4d} | Map {map_seed:3d} | Reward: {reward:7.1f} | Epsilon: {agent.epsilon:.4f} | Ticks: {agent.last_episode_ticks}")

    if (episode + 1) % EVAL_EVERY == 0:
        eval_reward, eval_ticks = agent.evaluate(TEST_SEEDS)
        eval_episodes.append(episode + 1)
        eval_rewards.append(eval_reward)
        print(f"  >> EVAL on {len(TEST_SEEDS)} held-out maps | Reward: {eval_reward:7.1f} | Ticks: {eval_ticks:7.1f}")
        # Checkpoint the best generalizing agent, judged on held-out maps (not the
        # noisy training reward, which is inflated by exploration on memorized seeds).
        agent.save_if_best(eval_reward, path="checkpoint_v3_best")

    if (episode + 1) % 50 == 0:
        agent.save()  # latest


# ---- summary plot ----
# fig, (ax_reward, ax_ticks) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
episodes = np.arange(1, len(rewards) + 1)

rewards = np.clip(rewards, -2000, 20000)
ticks = np.clip(ticks, -2000, 20000)

plt.subplots(figsize=(20, 10))
plt.plot(episodes, rewards, color="#39a6f5", linewidth=0.7, label="reward")
if len(rewards) >= ROLLING_WINDOW:
    rolling = np.convolve(rewards, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
    plt.plot(
        np.arange(ROLLING_WINDOW, len(rewards) + 1),
        rolling,
        color="#1f77b4",
        linewidth=2,
        label=f"rolling mean ({ROLLING_WINDOW})",
    )
# ax_reward.set_ylabel("episode reward")
# ax_reward.set_xlabel("episode")
# ax_reward.axhline(0, color="black", linewidth=0.5, alpha=0.3)
# ax_reward.legend(loc="best")
# ax_reward.grid(alpha=0.3)

# fig.suptitle(f"Training summary ({NUM_EPISODES} episodes, {agent.num_stations} stations)")
# fig.tight_layout()
# fig.savefig("training_summary.png", dpi=120)

plt.plot(episodes, ticks, color="#ee637a", linewidth=0.7, label="ticks survived")
if len(rewards) >= ROLLING_WINDOW:
    rolling = np.convolve(ticks, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
    plt.plot(
        np.arange(ROLLING_WINDOW, len(ticks) + 1),
        rolling,
        color="#e61e3f",
        linewidth=2,
        label=f"rolling mean ({ROLLING_WINDOW})",
    )
# Held-out evaluation curve (greedy, never-trained maps) — the generalization signal.
if eval_rewards:
    plt.plot(
        eval_episodes,
        np.clip(eval_rewards, -2000, 20000),
        color="#2ca02c",
        marker="o",
        linewidth=2,
        label="held-out eval reward",
    )

plt.ylabel("rewards/ticks")
plt.xlabel("episode")
# plt.axhline(0, color="black", linewidth=0.5, alpha=0.3)
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.yticks()
plt.suptitle(f"Training summary ({NUM_EPISODES} episodes, {agent.num_stations} stations)")
# plt.tight_layout()
plt.savefig("training_summary.png", dpi=120)

print("Saved training_summary.png")
plt.show()


# ---- final visualized run ----
with open("config.yaml") as f:
    config = yaml.safe_load(f)
WIDTH, HEIGHT = config["resolution"]

vis = Visualizer(WIDTH, HEIGHT)

vis.episode = NUM_EPISODES + 1
agent.epsilon = 0
agent.mcts_sims = 100  # deeper search for the showcase run (slower but stronger planning)
reward = agent.train_episode(visualizer=vis)
print(f"Final Run | Reward: {reward:7.1f} | Epsilon: {agent.epsilon:.4f}")

vis.hold()
vis.close()
