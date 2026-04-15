"""
MCTS (蒙特卡洛树搜索) 智能体模板。

学生作业：完成 MCTS 算法的核心实现。
参考：《深度学习与围棋》第 4 章
"""

import math
import random

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

__all__ = ["MCTSAgent"]


def _count_empty_points(board):
    empty_points = 0
    for row in range(1, board.num_rows + 1):
        for col in range(1, board.num_cols + 1):
            if board.get(Point(row, col)) is None:
                empty_points += 1
    return empty_points


def _has_adjacent_stone(game_state, point):
    for neighbor in point.neighbors():
        if (
            game_state.board.is_on_grid(neighbor)
            and game_state.board.get(neighbor) is not None
        ):
            return True
    return False


def _should_consider_pass(game_state):
    if game_state.last_move is not None and game_state.last_move.is_pass:
        return True
    return _count_empty_points(game_state.board) <= max(8, game_state.board.num_rows + 2)


def _search_moves(game_state):
    play_moves = _play_moves(game_state)
    if not play_moves:
        return [Move.pass_turn()]
    if _should_consider_pass(game_state):
        return play_moves + [Move.pass_turn()]
    return play_moves


def _play_moves(game_state):
    return [move for move in game_state.legal_moves() if move.is_play]


def _count_captured_stones(game_state, move):
    if not move.is_play:
        return 0
    captured = 0
    seen_strings = set()
    for neighbor in move.point.neighbors():
        if not game_state.board.is_on_grid(neighbor):
            continue
        neighbor_string = game_state.board.get_go_string(neighbor)
        if (
            neighbor_string is None
            or neighbor_string.color == game_state.next_player
        ):
            continue
        string_id = id(neighbor_string)
        if string_id in seen_strings:
            continue
        seen_strings.add(string_id)
        if neighbor_string.num_liberties == 1:
            captured += len(neighbor_string.stones)
    return captured


def _count_neighbor_stones(game_state, point, player):
    count = 0
    for neighbor in point.neighbors():
        if (
            game_state.board.is_on_grid(neighbor)
            and game_state.board.get(neighbor) == player
        ):
            count += 1
    return count


def _move_prior(game_state, move):
    if move.is_pass:
        return 0.25

    return max(0.1, 0.8 + _move_score(game_state, move) * 0.18)


def _move_score(game_state, move):
    center_row = (game_state.board.num_rows + 1) / 2.0
    center_col = (game_state.board.num_cols + 1) / 2.0
    capture_bonus = _count_captured_stones(game_state, move)
    connection_bonus = _count_neighbor_stones(
        game_state, move.point, game_state.next_player
    )
    center_distance = abs(move.point.row - center_row) + abs(
        move.point.col - center_col
    )
    local_bonus = 0.8 if _has_adjacent_stone(game_state, move.point) else 0.0
    next_state = game_state.apply_move(move)
    new_string = next_state.board.get_go_string(move.point)
    liberty_bonus = new_string.num_liberties if new_string is not None else 0
    return (
        capture_bonus * 5.0
        + liberty_bonus * 1.2
        + connection_bonus * 0.8
        + local_bonus
        - center_distance * 0.15
    )


def _score_lead(game_state, player):
    game_result = compute_game_result(game_state)
    lead = game_result.b - (game_result.w + game_result.komi)
    return lead if player == Player.black else -lead


class MCTSNode:
    """
    MCTS 树节点。

    属性：
        game_state: 当前局面
        parent: 父节点（None 表示根节点）
        children: 子节点列表
        visit_count: 访问次数
        value_sum: 累积价值（胜场数）
        prior: 先验概率（来自策略网络，可选）
    """

    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.unexpanded_moves = _search_moves(game_state) if not game_state.is_over() else []

    @property
    def value(self):
        """计算平均价值 = value_sum / visit_count，防止除零。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否为叶节点（未展开）。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def best_child(self, c=1.414):
        """
        选择最佳子节点（UCT 算法）。

        UCT = value + c * sqrt(ln(parent_visits) / visits)

        Args:
            c: 探索常数（默认 sqrt(2)）

        Returns:
            最佳子节点
        """
        if not self.children:
            raise ValueError("best_child() 需要在非叶节点上调用")

        log_parent_visits = math.log(max(1, self.visit_count))

        def uct_score(child):
            if child.visit_count == 0:
                return float("inf")
            exploitation = 1.0 - child.value
            exploration = c * child.prior * math.sqrt(
                log_parent_visits / child.visit_count
            )
            return exploitation + exploration

        return max(self.children, key=uct_score)

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        if self.is_terminal():
            return self
        if not self.children:
            for move in self.unexpanded_moves:
                next_state = self.game_state.apply_move(move)
                self.children.append(
                    MCTSNode(
                        next_state,
                        parent=self,
                        move=move,
                        prior=_move_prior(self.game_state, move),
                    )
                )
            self.unexpanded_moves = []
        if not self.children:
            return self
        return random.choices(
            self.children,
            weights=[child.prior for child in self.children],
            k=1,
        )[0]

    def backup(self, value):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            value: 从当前局面模拟得到的结果（1=胜，0=负，0.5=和）
        """
        node = self
        current_value = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            current_value = 1.0 - current_value
            node = node.parent


class MCTSAgent:
    """
    MCTS 智能体。

    属性：
        num_rounds: 每次决策的模拟轮数
        temperature: 温度参数（控制探索程度）
    """

    def __init__(
        self,
        num_rounds=1000,
        temperature=1.0,
        exploration_weight=1.414,
        rollout_depth=28,
    ):
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        流程：
            1. 创建根节点
            2. 进行 num_rounds 轮模拟：
               a. Selection: 用 UCT 选择路径到叶节点
               b. Expansion: 展开叶节点
               c. Simulation: 随机模拟至终局
               d. Backup: 反向传播结果
            3. 选择访问次数最多的棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        if self._should_pass_now(game_state) or self._should_force_pass(game_state):
            return Move.pass_turn()

        legal_moves = _search_moves(game_state)
        if not legal_moves:
            return Move.resign()
        if len(legal_moves) == 1:
            return legal_moves[0]

        root = MCTSNode(game_state)

        for _ in range(self.num_rounds):
            node = root
            while not node.is_leaf() and not node.is_terminal():
                node = node.best_child(self.exploration_weight)

            if not node.is_terminal():
                node = node.expand()

            value = self._simulate(node.game_state)
            node.backup(value)

        return self._select_best_move(root)

    def _simulate(self, game_state):
        """
        快速模拟（Rollout）：随机走子至终局。

        【第二小问要求】
        标准 MCTS 使用完全随机走子，但需要实现至少两种优化方法：
        1. 启发式走子策略（如：优先选有气、不自杀、提子的走法）
        2. 限制模拟深度（如：最多走 20-30 步后停止评估）
        3. 其他：快速走子评估（RAVE）、池势启发等

        Args:
            game_state: 起始局面

        Returns:
            从当前玩家视角的结果（1=胜, 0=负, 0.5=和）
        """
        root_player = game_state.next_player
        rollout_state = game_state

        for _ in range(self.rollout_depth):
            if rollout_state.is_over():
                return self._terminal_value(rollout_state, root_player)
            rollout_move = self._select_rollout_move(rollout_state)
            rollout_state = rollout_state.apply_move(rollout_move)

        return self._heuristic_rollout_value(rollout_state, root_player)

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        children = [child for child in root.children if child.move is not None]
        if not children:
            fallback_moves = _search_moves(root.game_state)
            return fallback_moves[0] if fallback_moves else Move.resign()

        if self.temperature != 1.0 and len(children) > 1:
            weights = [
                max(child.visit_count, 1) ** (1.0 / max(self.temperature, 1e-6))
                for child in children
            ]
            return random.choices(children, weights=weights, k=1)[0].move

        best_child = max(
            children,
            key=lambda child: (child.visit_count, 1.0 - child.value),
        )
        return best_child.move

    def _select_rollout_move(self, game_state):
        if self._should_pass_now(game_state):
            return Move.pass_turn()

        play_moves = [move for move in _search_moves(game_state) if move.is_play]
        if not play_moves:
            return Move.pass_turn()

        local_moves = [
            move for move in play_moves if _has_adjacent_stone(game_state, move.point)
        ]
        if local_moves:
            play_moves = local_moves

        center_row = (game_state.board.num_rows + 1) / 2.0
        center_col = (game_state.board.num_cols + 1) / 2.0
        move_scores = []

        for move in play_moves:
            score = _move_score(game_state, move)
            center_distance = abs(move.point.row - center_row) + abs(
                move.point.col - center_col
            )
            score -= center_distance * 0.02
            move_scores.append((score, move))

        best_score = max(score for score, _ in move_scores)
        candidates = [
            move for score, move in move_scores if score >= best_score - 0.75
        ]
        return random.choice(candidates)

    def _terminal_value(self, game_state, perspective_player):
        winner = game_state.winner()
        if winner is None:
            return 0.5
        return 1.0 if winner == perspective_player else 0.0

    def _should_pass_now(self, game_state):
        empty_points = _count_empty_points(game_state.board)
        if game_state.last_move is not None and game_state.last_move.is_pass:
            return (
                _score_lead(game_state, game_state.next_player) >= -1.0
                or self._best_play_score(game_state) <= 2.2
            )
        if empty_points > max(8, game_state.board.num_rows + 2):
            return False

        play_moves = _play_moves(game_state)
        if not play_moves:
            return True
        has_capture = any(
            _count_captured_stones(game_state, move) > 0 for move in play_moves
        )
        if has_capture:
            return False
        best_score = self._best_play_score(game_state)
        lead = _score_lead(game_state, game_state.next_player)

        if empty_points <= 3 and best_score <= 2.8:
            return True
        if empty_points <= 5 and lead > 0.0 and best_score <= 2.4:
            return True
        if empty_points <= 7 and lead > 2.0 and best_score <= 1.8:
            return True
        return False

    def _should_force_pass(self, game_state):
        board_points = game_state.board.num_rows * game_state.board.num_cols
        move_count = 0
        state = game_state
        while state.previous_state is not None and move_count <= board_points * 2:
            move_count += 1
            state = state.previous_state

        if move_count < board_points + 6:
            return False
        empty_points = _count_empty_points(game_state.board)
        if empty_points > 8:
            return False
        if self._best_play_score(game_state) > 2.6:
            return False
        return True

    def _best_play_score(self, game_state):
        play_moves = _play_moves(game_state)
        if not play_moves:
            return float("-inf")
        return max(_move_score(game_state, move) for move in play_moves)

    def _heuristic_rollout_value(self, game_state, perspective_player):
        board = game_state.board
        my_stones = 0
        opp_stones = 0
        my_liberties = 0
        opp_liberties = 0
        visited_strings = set()

        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                point = Point(row, col)
                string = board.get_go_string(point)
                if string is None:
                    continue
                string_id = id(string)
                if string_id in visited_strings:
                    continue
                visited_strings.add(string_id)

                if string.color == perspective_player:
                    my_stones += len(string.stones)
                    my_liberties += string.num_liberties
                else:
                    opp_stones += len(string.stones)
                    opp_liberties += string.num_liberties

        game_result = compute_game_result(game_state)
        territory_diff = game_result.b - (game_result.w + game_result.komi)
        if perspective_player == Player.white:
            territory_diff = -territory_diff

        feature_score = (
            territory_diff * 2.0
            + (my_stones - opp_stones) * 1.5
            + (my_liberties - opp_liberties) * 0.35
        )
        return 0.5 + 0.45 * math.tanh(feature_score / 10.0)
