"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

import math

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

__all__ = ["MinimaxAgent"]



class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """

    def __init__(self, max_depth=3, evaluator=None):
        self.max_depth = max_depth
        # 默认评估函数（TODO：学生可替换为神经网络）
        self.evaluator = evaluator or self._default_evaluator
        self.cache = GameResultCache()
        self._root_player = Player.black

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        self._root_player = game_state.next_player
        self.cache = GameResultCache()
        if self._should_pass_now(game_state):
            return Move.pass_turn()

        best_score = -math.inf
        best_move = None

        for move in self._get_ordered_moves(game_state):
            if move.is_resign:
                continue
            next_state = game_state.apply_move(move)
            score = self.alphabeta(
                next_state,
                self.max_depth - 1,
                -math.inf,
                math.inf,
                maximizing_player=False,
            )
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move is not None else Move.pass_turn()

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        if depth <= 0 or game_state.is_over():
            return self.evaluator(game_state)

        moves = [move for move in self._get_ordered_moves(game_state) if not move.is_resign]
        if not moves:
            return self.evaluator(game_state)

        if maximizing_player:
            value = -math.inf
            for move in moves:
                value = max(
                    value,
                    self.minimax(game_state.apply_move(move), depth - 1, False),
                )
            return value

        value = math.inf
        for move in moves:
            value = min(
                value,
                self.minimax(game_state.apply_move(move), depth - 1, True),
            )
        return value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        cache_key = (game_state.next_player, game_state.board.zobrist_hash())
        cached = self.cache.get(cache_key)
        if cached is not None and cached["depth"] >= depth:
            return cached["value"]

        if depth <= 0 or game_state.is_over():
            value = self.evaluator(game_state)
            self.cache.put(cache_key, depth, value)
            return value

        moves = [move for move in self._get_ordered_moves(game_state) if not move.is_resign]
        if not moves:
            value = self.evaluator(game_state)
            self.cache.put(cache_key, depth, value)
            return value

        if maximizing_player:
            value = -math.inf
            for move in moves:
                value = max(
                    value,
                    self.alphabeta(
                        game_state.apply_move(move),
                        depth - 1,
                        alpha,
                        beta,
                        False,
                    ),
                )
                alpha = max(alpha, value)
                if value >= beta:
                    break
        else:
            value = math.inf
            for move in moves:
                value = min(
                    value,
                    self.alphabeta(
                        game_state.apply_move(move),
                        depth - 1,
                        alpha,
                        beta,
                        True,
                    ),
                )
                beta = min(beta, value)
                if value <= alpha:
                    break

        self.cache.put(cache_key, depth, value)
        return value

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        if game_state.is_over():
            winner = game_state.winner()
            if winner is None:
                return 0.0
            return 10_000.0 if winner == self._root_player else -10_000.0

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

                stone_count = len(string.stones)
                liberty_count = string.num_liberties
                if string.color == self._root_player:
                    my_stones += stone_count
                    my_liberties += liberty_count
                else:
                    opp_stones += stone_count
                    opp_liberties += liberty_count

        game_result = compute_game_result(game_state)
        territory_diff = game_result.b - (game_result.w + game_result.komi)
        if self._root_player == Player.white:
            territory_diff = -territory_diff

        stone_diff = my_stones - opp_stones
        liberty_diff = my_liberties - opp_liberties
        return territory_diff * 3.0 + stone_diff * 1.5 + liberty_diff * 0.4

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        moves = [move for move in game_state.legal_moves() if not move.is_resign]
        return sorted(moves, key=lambda move: self._move_score(game_state, move), reverse=True)

    def _count_captured_stones(self, game_state, move):
        if not move.is_play:
            return 0
        captured = 0
        seen_strings = set()
        for neighbor in move.point.neighbors():
            if not game_state.board.is_on_grid(neighbor):
                continue
            string = game_state.board.get_go_string(neighbor)
            if string is None or string.color == game_state.next_player:
                continue
            string_id = id(string)
            if string_id in seen_strings:
                continue
            seen_strings.add(string_id)
            if string.num_liberties == 1:
                captured += len(string.stones)
        return captured

    def _move_score(self, game_state, move):
        if move.is_pass:
            empty_points = self._count_empty_points(game_state)
            if empty_points <= 4:
                return 2.0
            return -1_000.0

        next_state = game_state.apply_move(move)
        new_string = next_state.board.get_go_string(move.point)
        liberties = new_string.num_liberties if new_string is not None else 0
        capture_bonus = self._count_captured_stones(game_state, move)
        return capture_bonus * 10.0 + liberties

    def _count_empty_points(self, game_state):
        empty_points = 0
        for row in range(1, game_state.board.num_rows + 1):
            for col in range(1, game_state.board.num_cols + 1):
                if game_state.board.get(Point(row, col)) is None:
                    empty_points += 1
        return empty_points

    def _score_lead(self, game_state):
        game_result = compute_game_result(game_state)
        lead = game_result.b - (game_result.w + game_result.komi)
        return lead if self._root_player == Player.black else -lead

    def _should_pass_now(self, game_state):
        empty_points = self._count_empty_points(game_state)
        if empty_points > 5:
            return False

        play_moves = [move for move in game_state.legal_moves() if move.is_play]
        if not play_moves:
            return True

        has_capture = any(
            self._count_captured_stones(game_state, move) > 0 for move in play_moves
        )
        if has_capture:
            return False

        if game_state.last_move is not None and game_state.last_move.is_pass:
            return self._score_lead(game_state) >= -1.0

        best_local_score = max(self._move_score(game_state, move) for move in play_moves)
        return self._score_lead(game_state) > 0.0 and best_local_score <= 2.0



class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag='exact'):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        current = self.cache.get(zobrist_hash)
        if current is None or depth >= current["depth"]:
            self.cache[zobrist_hash] = {
                "depth": depth,
                "value": value,
                "flag": flag,
            }
