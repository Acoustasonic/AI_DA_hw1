import threading
import tkinter as tk
from tkinter import ttk

from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from dlgo import GameState, Player, Point, compute_game_result
from dlgo.goboard import Move


class GoApp(tk.Tk):
    BOARD_PIXELS = 520
    BOARD_PADDING = 40
    STONE_RATIO = 0.42

    def __init__(self):
        super().__init__()
        self.title("围棋 AI 作业演示")
        self.resizable(False, False)

        self.board_size_var = tk.IntVar(value=5)
        self.black_controller_var = tk.StringVar(value="human")
        self.white_controller_var = tk.StringVar(value="mcts")
        self.ai_delay_var = tk.IntVar(value=300)
        self.status_var = tk.StringVar()
        self.score_var = tk.StringVar()

        self.game_state = GameState.new_game(self.board_size_var.get())
        self.waiting_for_ai = False
        self._request_token = 0

        self._build_ui()
        self.new_game()

    def _build_ui(self):
        self.configure(bg="#f3eadb")

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="ew")

        ttk.Label(controls, text="棋盘大小").grid(row=0, column=0, padx=4)
        ttk.Combobox(
            controls,
            textvariable=self.board_size_var,
            values=(5, 9),
            width=5,
            state="readonly",
        ).grid(row=0, column=1, padx=4)

        ttk.Label(controls, text="黑方").grid(row=0, column=2, padx=4)
        ttk.Combobox(
            controls,
            textvariable=self.black_controller_var,
            values=("human", "random", "mcts", "minimax"),
            width=10,
            state="readonly",
        ).grid(row=0, column=3, padx=4)

        ttk.Label(controls, text="白方").grid(row=0, column=4, padx=4)
        ttk.Combobox(
            controls,
            textvariable=self.white_controller_var,
            values=("human", "random", "mcts", "minimax"),
            width=10,
            state="readonly",
        ).grid(row=0, column=5, padx=4)

        ttk.Label(controls, text="AI延迟(ms)").grid(row=0, column=6, padx=4)
        ttk.Combobox(
            controls,
            textvariable=self.ai_delay_var,
            values=(0, 150, 300, 500, 800),
            width=8,
            state="readonly",
        ).grid(row=0, column=7, padx=4)

        ttk.Button(controls, text="新游戏", command=self.new_game).grid(
            row=0, column=8, padx=6
        )
        ttk.Button(controls, text="停一手", command=self.pass_turn).grid(
            row=0, column=9, padx=6
        )

        self.canvas = tk.Canvas(
            self,
            width=self.BOARD_PIXELS,
            height=self.BOARD_PIXELS,
            bg="#cfa36a",
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=0, padx=12, pady=(0, 8))
        self.canvas.bind("<Button-1>", self._handle_click)

        info = ttk.Frame(self, padding=(12, 0, 12, 12))
        info.grid(row=2, column=0, sticky="ew")

        ttk.Label(info, textvariable=self.status_var, font=("Microsoft YaHei UI", 11)).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(info, textvariable=self.score_var).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def new_game(self):
        self._request_token += 1
        self.waiting_for_ai = False
        self.game_state = GameState.new_game(self.board_size_var.get())
        self._redraw()
        self._maybe_start_ai_turn()

    def pass_turn(self):
        if self.waiting_for_ai or self.game_state.is_over():
            return
        if not self._is_human_turn():
            return
        self._apply_move(Move.pass_turn())

    def _build_agent(self, controller_name):
        if controller_name == "random":
            return RandomAgent()
        if controller_name == "minimax":
            return MinimaxAgent(max_depth=2)
        return MCTSAgent(num_rounds=60, rollout_depth=20)

    def _controller_for(self, player):
        if player == Player.black:
            return self.black_controller_var.get()
        return self.white_controller_var.get()

    def _is_human_turn(self):
        return self._controller_for(self.game_state.next_player) == "human"

    def _player_label(self, player):
        return "黑棋" if player == Player.black else "白棋"

    def _maybe_start_ai_turn(self):
        if self.game_state.is_over() or self.waiting_for_ai:
            self._update_status()
            return
        if self._is_human_turn():
            self._update_status()
            return

        self.waiting_for_ai = True
        self._update_status()
        token = self._request_token
        state = self.game_state
        agent = self._build_agent(self._controller_for(state.next_player))
        threading.Thread(
            target=self._run_ai_move,
            args=(token, state, agent),
            daemon=True,
        ).start()

    def _run_ai_move(self, token, state, agent):
        move = agent.select_move(state)
        self.after(0, lambda: self._finish_ai_move(token, move))

    def _finish_ai_move(self, token, move):
        if token != self._request_token:
            return
        self.waiting_for_ai = False
        self._apply_move(move)

    def _handle_click(self, event):
        if self.waiting_for_ai or self.game_state.is_over():
            return
        if not self._is_human_turn():
            return

        point = self._event_to_point(event.x, event.y)
        if point is None:
            return

        move = Move.play(point)
        if not self.game_state.is_valid_move(move):
            self.status_var.set("该位置不能落子，请换一个点。")
            return

        self._apply_move(move)

    def _event_to_point(self, x, y):
        board_size = self.board_size_var.get()
        spacing = self._spacing()
        row = round((y - self.BOARD_PADDING) / spacing) + 1
        col = round((x - self.BOARD_PADDING) / spacing) + 1

        if not (1 <= row <= board_size and 1 <= col <= board_size):
            return None

        px, py = self._point_to_canvas(Point(row, col))
        if abs(px - x) > spacing * 0.45 or abs(py - y) > spacing * 0.45:
            return None
        return Point(row, col)

    def _apply_move(self, move):
        if not self.game_state.is_valid_move(move):
            return
        self.game_state = self.game_state.apply_move(move)
        self._redraw()
        if not self.game_state.is_over():
            self.after(self.ai_delay_var.get(), self._maybe_start_ai_turn)

    def _spacing(self):
        board_size = self.board_size_var.get()
        if board_size == 1:
            return 0
        return (self.BOARD_PIXELS - 2 * self.BOARD_PADDING) / (board_size - 1)

    def _point_to_canvas(self, point):
        spacing = self._spacing()
        x = self.BOARD_PADDING + (point.col - 1) * spacing
        y = self.BOARD_PADDING + (point.row - 1) * spacing
        return x, y

    def _redraw(self):
        self.canvas.delete("all")
        board_size = self.board_size_var.get()
        spacing = self._spacing()

        self.canvas.create_rectangle(
            0,
            0,
            self.BOARD_PIXELS,
            self.BOARD_PIXELS,
            fill="#d9ae74",
            outline="",
        )

        for idx in range(board_size):
            offset = self.BOARD_PADDING + idx * spacing
            self.canvas.create_line(
                self.BOARD_PADDING,
                offset,
                self.BOARD_PIXELS - self.BOARD_PADDING,
                offset,
                fill="#5b3920",
                width=2,
            )
            self.canvas.create_line(
                offset,
                self.BOARD_PADDING,
                offset,
                self.BOARD_PIXELS - self.BOARD_PADDING,
                fill="#5b3920",
                width=2,
            )

        if board_size % 2 == 1:
            center = (board_size + 1) // 2
            x, y = self._point_to_canvas(Point(center, center))
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#5b3920", outline="")

        radius = spacing * self.STONE_RATIO
        for row in range(1, board_size + 1):
            for col in range(1, board_size + 1):
                point = Point(row, col)
                stone = self.game_state.board.get(point)
                if stone is None:
                    continue
                x, y = self._point_to_canvas(point)
                fill = "#111111" if stone == Player.black else "#f6f4ef"
                outline = "#111111" if stone == Player.black else "#c9c3b6"
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline=outline,
                    width=2,
                )

        if self.game_state.last_move is not None and self.game_state.last_move.is_play:
            x, y = self._point_to_canvas(self.game_state.last_move.point)
            self.canvas.create_rectangle(
                x - 5,
                y - 5,
                x + 5,
                y + 5,
                outline="#d64545",
                width=2,
            )

        self._update_status()

    def _update_status(self):
        score = compute_game_result(self.game_state)
        score_text = f"当前估分: 黑 {score.b:.1f} / 白 {score.w + score.komi:.1f}"
        self.score_var.set(score_text)

        if self.game_state.is_over():
            winner = self.game_state.winner()
            if winner is None:
                self.status_var.set("对局结束：平局。")
            else:
                self.status_var.set(f"对局结束：{self._player_label(winner)} 获胜。")
            return

        player_name = self._player_label(self.game_state.next_player)
        controller_name = self._controller_for(self.game_state.next_player)
        if self.waiting_for_ai:
            self.status_var.set(f"{player_name}({controller_name}) 思考中...")
        elif self._is_human_turn():
            self.status_var.set(f"{player_name} 轮到你落子。")
        else:
            self.status_var.set(f"{player_name}({controller_name}) 准备落子。")


if __name__ == "__main__":
    app = GoApp()
    app.mainloop()
