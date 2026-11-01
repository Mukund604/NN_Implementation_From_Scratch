"""Classic AI: heuristic search, adversarial search and reinforcement learning."""
import heapq

import numpy as np


def astar(grid, start, goal):
    """Shortest 4 connected path on a grid where 1 marks a wall, or None."""
    rows, cols = len(grid), len(grid[0])

    def h(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    frontier = [(h(start), 0, start)]
    came_from, best = {start: None}, {start: 0}
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1]
        if cost > best[cell]:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or grid[nxt[0]][nxt[1]]:
                continue
            if cost + 1 < best.get(nxt, float("inf")):
                best[nxt], came_from[nxt] = cost + 1, cell
                heapq.heappush(frontier, (cost + 1 + h(nxt), cost + 1, nxt))
    return None


WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(board):
    for a, b, c in WINS:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def alphabeta(board, player, alpha=-2, beta=2):
    """Tic tac toe minimax with alpha beta pruning. Returns (value, move) for player 1 as max."""
    won = winner(board)
    if won:
        return won, None
    moves = [i for i, v in enumerate(board) if v == 0]
    if not moves:
        return 0, None
    best_move = moves[0]
    if player == 1:
        value = -2
        for m in moves:
            board[m] = 1
            score, _ = alphabeta(board, -1, alpha, beta)
            board[m] = 0
            if score > value:
                value, best_move = score, m
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = 2
        for m in moves:
            board[m] = -1
            score, _ = alphabeta(board, 1, alpha, beta)
            board[m] = 0
            if score < value:
                value, best_move = score, m
            beta = min(beta, value)
            if alpha >= beta:
                break
    return value, best_move


class CliffWalking:
    """The 4 by 12 cliff walking task from Sutton and Barto."""

    rows, cols = 4, 12
    start, goal = (3, 0), (3, 11)
    n_actions = 4

    def step(self, state, action):
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
        r = min(max(state[0] + dr, 0), self.rows - 1)
        c = min(max(state[1] + dc, 0), self.cols - 1)
        if r == 3 and 0 < c < 11:
            return self.start, -100, False
        return (r, c), -1, (r, c) == self.goal


def q_learning(env, episodes=500, lr=0.5, gamma=1.0, epsilon=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.rows, env.cols, env.n_actions))
    returns = []
    for _ in range(episodes):
        state, done, total = env.start, False, 0
        while not done:
            explore = rng.random() < epsilon
            action = int(rng.integers(env.n_actions)) if explore else int(Q[state].argmax())
            nxt, reward, done = env.step(state, action)
            target = reward + (0 if done else gamma * Q[nxt].max())
            Q[state][action] += lr * (target - Q[state][action])
            state = nxt
            total += reward
        returns.append(total)
    return Q, returns


def greedy_path_length(env, Q, limit=200):
    state, steps = env.start, 0
    while steps < limit:
        state, _, done = env.step(state, int(Q[state].argmax()))
        steps += 1
        if done:
            return steps
    return None
