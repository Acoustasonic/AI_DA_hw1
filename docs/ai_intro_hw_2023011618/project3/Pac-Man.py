import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import argparse
import random


UNIT = 100
Map_H = 5
Map_W = 5
BEAN_POSITIONS = [(0, 2), (2, 3)]
GHOST_POSITIONS = [(1, 2), (2, 1), (3, 3)]
START_POS = (0, 0)
GOAL_POS = (4, 4)
ACTIONS = ['u', 'd', 'l', 'r']
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
STEP_REWARD = -1
WALL_REWARD = -10
BEAN_REWARD = 10
GHOST_REWARD = -100
GOAL_REWARD_ALL_BEANS = 100
GOAL_REWARD_MISSING_BEANS = -20

map_state = np.array([
    [ 0,  1,  2,  3,  4],
    [ 5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24]
])


def grid_to_canvas(col, row):
    origin = np.array([UNIT / 2, UNIT / 2])
    return origin[0] + col * UNIT, origin[1] + row * UNIT


class Map(tk.Tk, object):
    def __init__(self):
        super(Map, self).__init__()
        self.action_space = ACTIONS
        self.n_actions = len(self.action_space)
        self.title('Pac-Man')
        self.geometry('{0}x{1}+400+50'.format(Map_W * UNIT, Map_H * UNIT))

        # 2颗豆子 (row, col)
        self.bean_positions = list(BEAN_POSITIONS)
        # 3个静止幽灵 (row, col)
        self.ghosts = [{'row': row, 'col': col, 'type': 'static'} for row, col in GHOST_POSITIONS]
        self._build_map()

    def _build_map(self):
        self.canvas = tk.Canvas(self, bg='white', height=Map_H * UNIT, width=Map_W * UNIT)
        for c in range(0, Map_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, Map_H * UNIT)
        for r in range(0, Map_H * UNIT, UNIT):
            self.canvas.create_line(0, r, Map_W * UNIT, r)

        origin = np.array([UNIT / 2, UNIT / 2])
        IMG_SIZE = (80, 80)
        asset_dir = Path(__file__).resolve().parent

        def load_img(path):
            return ImageTk.PhotoImage(Image.open(path).resize(IMG_SIZE, Image.Resampling.LANCZOS))

        self.bm_beans = load_img(asset_dir / "beans.png")
        self.bm_ghost = load_img(asset_dir / "ghost.png")
        self.bm_person = load_img(asset_dir / "pac-man.png")
        self.bm_flag = load_img(asset_dir / "destination.png")

        # 终点 (4,4)
        self.flag = self.canvas.create_image(origin[0]+UNIT*4, origin[1]+UNIT*4,
                                              image=self.bm_flag, tag="destination")

        # 豆子
        self.bean_items = []
        for i, (row, col) in enumerate(self.bean_positions):
            cx, cy = grid_to_canvas(col, row)
            item = self.canvas.create_image(cx, cy, image=self.bm_beans, tag="bean%d" % i)
            self.bean_items.append(item)

        # 幽灵
        self.ghost_items = []
        for i, g in enumerate(self.ghosts):
            cx, cy = grid_to_canvas(g['col'], g['row'])
            item = self.canvas.create_image(cx, cy, image=self.bm_ghost, tag="ghost%d" % i)
            self.ghost_items.append(item)

        # 吃豆人，初始位置(0,0)
        cx, cy = grid_to_canvas(0, 0)
        self.person = self.canvas.create_image(cx, cy, image=self.bm_person)
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.person)
        origin = np.array([UNIT / 2, UNIT / 2])

        self.bean_positions = list(BEAN_POSITIONS)
        for item in self.bean_items:
            self.canvas.delete(item)
        self.bean_items = []
        for i, (row, col) in enumerate(self.bean_positions):
            cx, cy = grid_to_canvas(col, row)
            item = self.canvas.create_image(cx, cy, image=self.bm_beans, tag="bean%d" % i)
            self.bean_items.append(item)

        self.ghosts = [{'row': row, 'col': col, 'type': 'static'} for row, col in GHOST_POSITIONS]
        for i, g in enumerate(self.ghosts):
            self.canvas.delete(self.ghost_items[i])
            cx, cy = grid_to_canvas(g['col'], g['row'])
            item = self.canvas.create_image(cx, cy, image=self.bm_ghost, tag="ghost%d" % i)
            self.ghost_items[i] = item

        cx, cy = grid_to_canvas(0, 0)
        self.person = self.canvas.create_image(cx, cy, image=self.bm_person)
        self.render()
        return self.get_state()

    def get_state(self):
        coords = self.canvas.coords(self.person)
        col = int(coords[0] / UNIT)
        row = int(coords[1] / UNIT)
        bean_mask = 0
        for idx, position in enumerate(BEAN_POSITIONS):
            if position not in self.bean_positions:
                bean_mask |= 1 << idx
        return int(map_state[row, col]), bean_mask

    def _get_pacman_grid_pos(self):
        coords = self.canvas.coords(self.person)
        col = min(int(coords[0] / UNIT), Map_W - 1)
        row = min(int(coords[1] / UNIT), Map_H - 1)
        return row, col

    def _check_ghost_collision(self):
        row, col = self._get_pacman_grid_pos()
        for g in self.ghosts:
            if row == g['row'] and col == g['col']:
                return True
        return False

    def step(self, action):
        """执行一个动作
        action: 0=上, 1=下, 2=左, 3=右
        """
        s = self.canvas.coords(self.person)
        base_action = np.array([0, 0])
        cost = -1  # 每走一步基础代价

        if action == 0:  # 上
            if s[1] >= UNIT:
                base_action[1] -= UNIT
            else:
                cost = -10  # 碰壁惩罚
        elif action == 1:  # 下
            if s[1] < (Map_H - 1) * UNIT:
                base_action[1] += UNIT
            else:
                cost = -10
        elif action == 2:  # 左
            if s[0] >= UNIT:
                base_action[0] -= UNIT
            else:
                cost = -10
        elif action == 3:  # 右
            if s[0] < (Map_W - 1) * UNIT:
                base_action[0] += UNIT
            else:
                cost = -10

        self.canvas.move(self.person, base_action[0], base_action[1])
        row, col = self._get_pacman_grid_pos()

        if (row, col) in self.bean_positions:
            bean_idx = self.bean_positions.index((row, col))
            cost += BEAN_REWARD
            self.bean_positions.pop(bean_idx)
            self.canvas.delete(self.bean_items.pop(bean_idx))

        if self._check_ghost_collision():
            return self.get_state(), GHOST_REWARD, True

        if (row, col) == GOAL_POS:
            if self.bean_positions:
                cost += GOAL_REWARD_MISSING_BEANS
            else:
                cost += GOAL_REWARD_ALL_BEANS
            return self.get_state(), cost, True

        return self.get_state(), cost, False

    def render(self):
        time.sleep(0.1)
        self.update()
        time.sleep(0.1)

def initial_model_state():
    return START_POS[0], START_POS[1], 0


def all_model_states():
    for row in range(Map_H):
        for col in range(Map_W):
            if (row, col) in GHOST_POSITIONS:
                continue
            for bean_mask in range(1 << len(BEAN_POSITIONS)):
                yield row, col, bean_mask


def transition(state, action):
    row, col, bean_mask = state
    dr, dc = ACTION_DELTAS[action]
    next_row, next_col = row + dr, col + dc
    reward = STEP_REWARD

    if not (0 <= next_row < Map_H and 0 <= next_col < Map_W):
        next_row, next_col = row, col
        reward = WALL_REWARD

    if (next_row, next_col) in GHOST_POSITIONS:
        return (next_row, next_col, bean_mask), GHOST_REWARD, True

    next_mask = bean_mask
    if (next_row, next_col) in BEAN_POSITIONS:
        idx = BEAN_POSITIONS.index((next_row, next_col))
        if not (bean_mask & (1 << idx)):
            next_mask |= (1 << idx)
            reward += BEAN_REWARD

    if (next_row, next_col) == GOAL_POS:
        if next_mask == (1 << len(BEAN_POSITIONS)) - 1:
            reward += GOAL_REWARD_ALL_BEANS
        else:
            reward += GOAL_REWARD_MISSING_BEANS
        return (next_row, next_col, next_mask), reward, True

    return (next_row, next_col, next_mask), reward, False


def value_iteration(gamma=0.9, theta=1e-8, max_iterations=10000):
    states = list(all_model_states())
    values = {state: 0.0 for state in states}
    policy = {}
    deltas = []

    for _ in range(max_iterations):
        delta = 0.0
        for state in states:
            row, col, _ = state
            if (row, col) == GOAL_POS:
                continue
            action_values = []
            for action in range(len(ACTIONS)):
                next_state, reward, done = transition(state, action)
                action_values.append(reward if done else reward + gamma * values[next_state])
            best_value = max(action_values)
            delta = max(delta, abs(values[state] - best_value))
            values[state] = best_value
            policy[state] = int(np.argmax(action_values))
        deltas.append(delta)
        if delta < theta:
            break
    return values, policy, deltas


def greedy_rollout(policy, max_steps=50):
    state = initial_model_state()
    trajectory = []
    total_reward = 0
    for _ in range(max_steps):
        action = policy.get(state, 0)
        next_state, reward, done = transition(state, action)
        trajectory.append((state, ACTIONS[action], reward, next_state))
        total_reward += reward
        state = next_state
        if done:
            break
    return trajectory, total_reward


def monte_carlo_control(episodes=5000, gamma=0.9, epsilon=0.2, min_epsilon=0.02, max_steps=80, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    q_values = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=float))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    episode_rewards = []

    def choose_action(state, current_epsilon):
        if random.random() < current_epsilon:
            return random.randrange(len(ACTIONS))
        return int(np.argmax(q_values[state]))

    for episode in range(episodes):
        state = initial_model_state()
        current_epsilon = max(min_epsilon, epsilon * (1 - episode / episodes))
        episode_data = []
        total_reward = 0

        for _ in range(max_steps):
            action = choose_action(state, current_epsilon)
            next_state, reward, done = transition(state, action)
            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state
            if done:
                break

        returns = []
        g = 0.0
        for state, action, reward in reversed(episode_data):
            g = gamma * g + reward
            returns.append((state, action, g))
        returns.reverse()

        visited = set()
        for state, action, g in returns:
            pair = (state, action)
            if pair in visited:
                continue
            visited.add(pair)
            returns_sum[pair] += g
            returns_count[pair] += 1
            q_values[state][action] = returns_sum[pair] / returns_count[pair]
        episode_rewards.append(total_reward)

    policy = {state: int(np.argmax(actions)) for state, actions in q_values.items()}
    return q_values, policy, episode_rewards


def print_policy_result(name, policy):
    trajectory, total_reward = greedy_rollout(policy)
    print(f"{name} total reward: {total_reward}")
    for idx, (state, action, reward, next_state) in enumerate(trajectory, 1):
        print(f"{idx:02d}. {state} --{action}/{reward}--> {next_state}")


def plot_learning_curves(vi_deltas, mc_rewards, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(vi_deltas)
    plt.title("Value iteration convergence")
    plt.xlabel("iteration")
    plt.ylabel("max value change")

    plt.subplot(1, 2, 2)
    window = 100
    if len(mc_rewards) >= window:
        smooth = np.convolve(mc_rewards, np.ones(window) / window, mode="valid")
        plt.plot(smooth)
        plt.ylabel(f"{window}-episode average reward")
    else:
        plt.plot(mc_rewards)
        plt.ylabel("episode reward")
    plt.title("Monte Carlo learning")
    plt.xlabel("episode")
    plt.tight_layout()
    chart_path = output_dir / "learning_curves.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    return chart_path


def run_cli(args):
    values, vi_policy, vi_deltas = value_iteration(gamma=args.gamma)
    print_policy_result("Value iteration", vi_policy)

    q_values, mc_policy, mc_rewards = monte_carlo_control(
        episodes=args.episodes,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    print_policy_result("Monte Carlo", mc_policy)
    chart_path = plot_learning_curves(vi_deltas, mc_rewards, Path(args.output_dir))
    print(f"Learning curves saved to: {chart_path}")

    if args.demo:
        env = Map()
        state = env.reset()
        for model_state, action_char, _, _ in greedy_rollout(vi_policy)[0]:
            action = ACTIONS.index(action_char)
            state, reward, done = env.step(action)
            env.render()
            if done:
                break
        env.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pac-Man reinforcement learning project")
    parser.add_argument("--demo", action="store_true", help="show Tkinter animation for the value-iteration policy")
    parser.add_argument("--episodes", type=int, default=5000, help="Monte Carlo training episodes")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="initial epsilon for Monte Carlo control")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--output-dir", default="../project3_outputs", help="directory for generated figures")
    run_cli(parser.parse_args())
