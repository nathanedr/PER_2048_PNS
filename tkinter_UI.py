import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import time
import threading
import json

from logic_3x3 import *
from logic_4x4 import *

def load_policy_and_value_from_json(filename='agent_state.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    policy = {int(k): v for k, v in data["policy"].items()}
    return policy

policy = load_policy_and_value_from_json("model_3x3_VI.json")


tdl = Learning()

tdl.add_feature(Pattern([0, 1, 2, 3, 4, 5]))
tdl.add_feature(Pattern([3, 4, 5, 6, 7, 8]))
tdl.add_feature(Pattern([0, 1, 3, 4, 6, 7]))
tdl.add_feature(Pattern([1, 2, 4, 5, 7, 8]))

tdl.load_gzip("test_model_3x3.bin.gz")

tdl_4x4 = Learning_4x4()

tdl_4x4.add_feature(Pattern_4x4([ 0, 1, 2, 3, 4, 5 ]))
tdl_4x4.add_feature(Pattern_4x4([ 4, 5, 6, 7, 8, 9 ]))
tdl_4x4.add_feature(Pattern_4x4([ 0, 1, 2, 4, 5, 6 ]))
tdl_4x4.add_feature(Pattern_4x4([ 4, 5, 6, 8, 9, 10 ]))

tdl_4x4.load_gzip("test_model_4x4.bin.gz")

BACKGROUND_COLOR = "#bbada0"
BOARD_COLOR = "#bbada0"
EMPTY_CELL_COLOR = "#cdc1b4"
TILE_COLORS = {
    2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", 32: "#f67c5f",
    64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", 512: "#edc850", 1024: "#edc53f",
    2048: "#edc22e", 4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32"
}
TILE_TEXT_COLORS = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2", 32: "#f9f6f2",
    64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2",
    2048: "#f9f6f2", 4096: "#f9f6f2", 8192: "#f9f6f2", 16384: "#f9f6f2"
}
CELL_SIZE = 100
CELL_PADDING = 10
BOARD_SIZE_3x3 = CELL_PADDING + (CELL_SIZE + CELL_PADDING) * 3
BOARD_SIZE_4x4 = CELL_PADDING + (CELL_SIZE + CELL_PADDING) * 4

class StartingPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BACKGROUND_COLOR)
        self.master = master
        self.pack(fill="both", expand=True)

        self.label = tk.Label(self, text="Projet Reinforcement Learning 2048", font=("Helvetica", 48, "bold"), bg=BACKGROUND_COLOR, fg="white")
        self.label.pack(pady=50)

        self.author = tk.Label(self, text="Nathan EDERY - Aubin COURTIAL - Alexis BLOND", font=("Helvetica", 18, "bold"), bg=BACKGROUND_COLOR, fg="white")
        self.author.pack(pady=20)

        self.button_3x3 = tk.Button(self, text="     3x3     ", font=("Helvetica", 24, "bold"),
                                    bg="#8f7a66", fg="white", command=self.start_3x3)
        self.button_3x3.pack(pady=20)

        self.button_4x4 = tk.Button(self, text="     4x4     ", font=("Helvetica", 24, "bold"),
                                    bg="#8f7a66", fg="white", command=self.start_4x4)
        self.button_4x4.pack(pady=20)

    def start_3x3(self):
        self.pack_forget()
        game_3x3 = Game2048GUI_3x3(self.master)
        game_3x3.pack(fill="both", expand=True)

    def start_4x4(self):
        self.pack_forget()
        game_4x4 = Game2048GUI_4x4(self.master)
        game_4x4.pack(fill="both", expand=True)

class Game2048GUI_3x3(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BACKGROUND_COLOR)
        self.master = master
        self.master.title("2048")
        self.pack(fill="both", expand=True, padx=20, pady=20)

        self.score = 0
        self.best_score = 0
        self.game_over = False
        self.win = False
        self.overlay_items = []
        self.simulating = False
        self.win_rank = 7 
        self.suggestion_visible = False
        self.suggestion_text = None
        self.last_move = None

        self.performance_texts = {
            "Random Agent": "2: 100.0%\n4: 100.0%\n8: 99.88%\n16: 95.42%\n32: 61.02%\n64: 9.1%\n128: 0.02%\n256: 0.0%\n512: 0.0%",
            "Value Iteration Agent": "2: 100.0%\n4: 100.0%\n8: 100.0%\n16: 99.99%\n32: 99.92%\n64: 99.2%\n128: 95.72%\n256: 82.91%\n512: 26.3%",
            "TDL Agent": "2: 100.0%\n4: 100.0%\n8: 100.0%\n16: 100.0%\n32: 100.0%\n64: 99.9%\n128: 98.8%\n256: 87.3%\n512: 29.0%"
        }

        self.controls_frame = tk.Frame(self, bg=BACKGROUND_COLOR)
        self.controls_frame.pack(pady=(0, 20), fill="x")
        self.board_frame = tk.Frame(self, bg=BACKGROUND_COLOR)
        self.board_frame.pack()

        self.header_frame = tk.Frame(self.controls_frame, bg=BACKGROUND_COLOR)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.header_frame.columnconfigure(0, weight=1)
        self.header_frame.columnconfigure(1, weight=1)
        self.header_frame.columnconfigure(2, weight=1)

        self.score_label = tk.Label(self.header_frame, text="Score: 0", font=("Helvetica", 20, "bold"),
                                    bg=BACKGROUND_COLOR, fg="white")
        self.score_label.grid(row=0, column=0, sticky="w", padx=10)
        self.best_label = tk.Label(self.header_frame, text="Best: 0", font=("Helvetica", 20, "bold"),
                                   bg=BACKGROUND_COLOR, fg="white")
        self.best_label.grid(row=0, column=1, sticky="w", padx=10)
        self.new_game_button = tk.Button(self.header_frame, text="New Game", font=("Helvetica", 18, "bold"),
                                         bg="#8f7a66", fg="white", bd=0, padx=15, pady=10,
                                         activebackground="#9f8b76", command=self.new_game)
        self.new_game_button.grid(row=0, column=2, sticky="e", padx=10)

        self.options_frame = tk.Frame(self.controls_frame, bg=BACKGROUND_COLOR)
        self.options_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.options_frame.columnconfigure(0, weight=1)
        self.options_frame.columnconfigure(1, weight=1)

        self.agent_label = tk.Label(self.options_frame, text="Agent", font=("Helvetica", 16, "bold"),
                                    bg=BACKGROUND_COLOR, fg="white")
        self.agent_label.grid(row=0, column=0, sticky="w", padx=10)
        self.agent_var = tk.StringVar(value="Random Agent")
        self.agent_menu = ttk.Combobox(self.options_frame, textvariable=self.agent_var,
                                       state="readonly", font=("Helvetica", 14))
        self.agent_menu['values'] = ("Random Agent", "Value Iteration Agent", "TDL Agent")
        self.agent_menu.grid(row=1, column=0, sticky="ew", padx=10)
        self.agent_menu.bind("<<ComboboxSelected>>", self.update_performance)
        # Prevent arrow keys from opening the dropdown
        self.agent_menu.bind("<Down>", lambda e: "break")
        self.agent_menu.bind("<Up>", lambda e: "break")
        self.agent_menu.bind("<Left>", lambda e: "break")
        self.agent_menu.bind("<Right>", lambda e: "break")
        # Return focus to canvas after selection
        self.agent_menu.bind("<<ComboboxSelected>>", lambda e: (self.update_performance(e), self.canvas.focus_set()))

        self.win_rank_label = tk.Label(self.options_frame, text="Winning Tile", font=("Helvetica", 16, "bold"),
                                       bg=BACKGROUND_COLOR, fg="white")
        self.win_rank_label.grid(row=0, column=1, sticky="w", padx=10)
        self.win_rank_var = tk.StringVar(value="128")
        self.win_rank_menu = ttk.Combobox(self.options_frame, textvariable=self.win_rank_var,
                                          state="readonly", font=("Helvetica", 14))
        self.win_rank_menu['values'] = ("16", "32", "64", "128", "256", "512")
        self.win_rank_menu.grid(row=1, column=1, sticky="ew", padx=10)
        self.win_rank_menu.bind("<<ComboboxSelected>>", self.update_win_rank)
        self.win_rank_menu.bind("<Down>", lambda e: "break")
        self.win_rank_menu.bind("<Up>", lambda e: "break")
        self.win_rank_menu.bind("<Left>", lambda e: "break")
        self.win_rank_menu.bind("<Right>", lambda e: "break")
        self.win_rank_menu.bind("<<ComboboxSelected>>", lambda e: (self.update_win_rank(e), self.canvas.focus_set()))

        self.controls_buttons_frame = tk.Frame(self.controls_frame, bg=BACKGROUND_COLOR)
        self.controls_buttons_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.controls_buttons_frame.columnconfigure(0, weight=1)
        self.controls_buttons_frame.columnconfigure(1, weight=1)
        
        self.simulate_button = tk.Button(self.controls_buttons_frame, text="Simulate Game",
                                         font=("Helvetica", 16, "bold"), bg="#8f7a66", fg="white",
                                         bd=0, padx=15, pady=10, activebackground="#9f8b76",
                                         command=self.toggle_simulation)
        self.simulate_button.grid(row=0, column=0, sticky="ew", padx=10)
        self.advice_button = tk.Button(self.controls_buttons_frame, text="Show Move Advice",
                                       font=("Helvetica", 16, "bold"), bg="#8f7a66", fg="white",
                                       bd=0, padx=15, pady=10, activebackground="#9f8b76",
                                       command=self.toggle_suggestion)
        self.advice_button.grid(row=0, column=1, sticky="ew", padx=10)

        self.slider_frame = tk.Frame(self.controls_frame, bg=BACKGROUND_COLOR)
        self.slider_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.slider_frame.columnconfigure(0, weight=1)
        self.slider_frame.columnconfigure(1, weight=1)
        self.delay_slider = tk.Scale(self.slider_frame, from_=0, to=500, orient="horizontal",
                                     label="Delay (ms)", resolution=1, length=250,
                                     bg=BACKGROUND_COLOR, fg="white", font=("Helvetica", 14),
                                     highlightthickness=0)
        self.delay_slider.set(10)
        self.delay_slider.grid(row=0, column=0, sticky="w", padx=10)
        self.move_time_label = tk.Label(self.slider_frame, text="Move Time: 0.0 ms",
                                        font=("Helvetica", 16, "bold"), bg=BACKGROUND_COLOR, fg="white")
        self.move_time_label.grid(row=0, column=1, sticky="e", padx=10)

        self.canvas = tk.Canvas(self.board_frame, width=BOARD_SIZE_3x3, height=BOARD_SIZE_3x3 + 40,
                                bg=BOARD_COLOR, highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.performance_frame = tk.LabelFrame(self.board_frame, text="Performance", font=("Helvetica", 14, "bold"),
                                               bg=BACKGROUND_COLOR, fg="white")
        self.performance_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        self.performance_text = tk.Text(self.performance_frame, width=20, height=10,
                                        bg=BACKGROUND_COLOR, fg="white", font=("Helvetica", 12),
                                        bd=0, highlightthickness=0)
        self.performance_text.pack(padx=10, pady=10)
        self.performance_text.config(state="disabled")

        self.back_button = tk.Button(self, text="Back", font=("Helvetica", 18, "bold"),
                                     bg="#8f7a66", fg="white", command=self.back_to_start)
        self.back_button.pack(pady=10)

        self.master.bind_all("<Key>", self.key_handler)
        self.new_game()
        self.update_performance()

    def update_performance(self, event=None):
        agent = self.agent_var.get()
        perf_text = self.performance_texts.get(agent, "")
        self.performance_text.config(state="normal")
        self.performance_text.delete("1.0", tk.END)
        lines = perf_text.strip().split("\n")
        for i, line in enumerate(lines):
            parts = line.split(":")
            if len(parts) < 2:
                self.performance_text.insert(tk.END, line+"\n")
                continue
            perc_str = parts[1].strip().replace("%", "")
            try:
                perc = float(perc_str)
            except:
                perc = 0.0
            red = int(255 * (100 - perc) / 100)
            green = int(255 * perc / 100)
            color = f"#{red:02x}{green:02x}00"
            tag_name = f"line{i}"
            self.performance_text.insert(tk.END, line+"\n", tag_name)
            self.performance_text.tag_config(tag_name, foreground=color)
        self.performance_text.config(state="disabled")
        self.canvas.focus_set()

    def update_win_rank(self, event):
        win_value = self.win_rank_var.get()
        self.win_rank = int(np.log2(int(win_value)))
        if get_max_rank_3x3(self.board) >= self.win_rank and not self.win:
            self.win = True
            self.show_overlay("You Win!")
            self.game_over = True
        self.canvas.focus_set()

    def update_suggestion(self):
        if self.game_over:
            return
        agent = self.agent_var.get()
        if agent == "Random Agent":
            valid_moves = [move for move in range(4) if execute_move_3x3(self.board, move) != self.board]
            action = random.choice(valid_moves) if valid_moves else None
        elif agent == "Value Iteration Agent":
            action = policy.get(self.board, None) if policy else None
        elif agent == "TDL Agent":
            action = tdl.select_best_move(Board(self.board)).opcode

            if action == 0:
                action = 0
            elif action == 1:
                action = 3
            elif action == 2:
                action = 1
            elif action == 3:
                action = 2
        else:
            action = None

        if self.suggestion_text:
            self.canvas.delete(self.suggestion_text)
            self.suggestion_text = None

        if action is not None:
            directions = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

            if action == -1:
                self.suggestion_text = self.canvas.create_text(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3 + 20,
                                                               text="No moves available",
                                                               fill="white", font=("Helvetica", 16, "bold"))
            else:
                suggestion = directions[action]
                self.suggestion_text = self.canvas.create_text(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3 + 20,
                                                            text=f"Suggested Move: {suggestion}",
                                                            fill="white", font=("Helvetica", 16, "bold"))
        else:
            self.suggestion_text = self.canvas.create_text(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3 + 20,
                                                           text="No moves available",
                                                           fill="white", font=("Helvetica", 16, "bold"))

    def toggle_suggestion(self):
        if self.suggestion_visible:
            if self.suggestion_text:
                self.canvas.delete(self.suggestion_text)
                self.suggestion_text = None
            self.suggestion_visible = False
            self.advice_button.config(text="Show Move Advice")
        else:
            self.suggestion_visible = True
            self.update_suggestion()
            self.advice_button.config(text="Hide Move Advice")

    def toggle_simulation(self):
        if self.simulating:
            self.simulating = False
            self.simulate_button.config(text="Simulate Game")
        else:
            self.simulating = True
            self.simulate_button.config(text="Stop Simulation")
            if self.agent_menu.get() == "Random Agent":
                threading.Thread(target=self.random_agent_play, daemon=True).start()
            elif self.agent_menu.get() == "Value Iteration Agent":
                threading.Thread(target=self.value_iteration_agent, daemon=True).start()
            elif self.agent_menu.get() == "TDL Agent":
                threading.Thread(target=self.TDL_agent_play, daemon=True).start()

    def random_agent_play(self):
        while self.simulating and not self.game_over:
            delay = self.delay_slider.get()
            time.sleep(delay / 1000.0)
            start = time.perf_counter()
            action = random.choice([0, 1, 2, 3])
            new_board, score, moved, win_flag, lose_flag = play_game_gui_3x3(
                self.board, action, self.win_rank, already_won=self.win)
            move_duration = (time.perf_counter() - start) * 1000.0
            self.canvas.after(0, lambda md=move_duration: self.move_time_label.config(text=f"Move Time: {md:.1f} ms"))
            if moved:
                self.board = new_board
                self.score += int(score)
                if self.score > self.best_score:
                    self.best_score = self.score
                self.update_gui()
                if win_flag and not self.win:
                    self.win = True
                    self.show_overlay("You Win!")
                    self.game_over = True
                elif lose_flag:
                    self.show_overlay("Game Over")
                    self.game_over = True
                    self.simulating = False
                    self.simulate_button.config(text="Simulate Game")

    def value_iteration_agent(self):
        while self.simulating and not self.game_over:
            delay = self.delay_slider.get()
            time.sleep(delay / 1000.0)
            start = time.perf_counter()
            action = policy.get(self.board, random.choice([0, 1, 2, 3])) if policy else random.choice([0, 1, 2, 3])
            new_board, score, moved, win_flag, lose_flag = play_game_gui_3x3(
                self.board, action, self.win_rank, already_won=self.win)
            move_duration = (time.perf_counter() - start) * 1000.0
            self.canvas.after(0, lambda md=move_duration: self.move_time_label.config(text=f"Move Time: {md:.1f} ms"))
            if moved:
                self.board = new_board
                self.score += int(score)
                if self.score > self.best_score:
                    self.best_score = self.score
                self.update_gui()
                if win_flag and not self.win:
                    self.win = True
                    self.show_overlay("You Win!")
                    self.game_over = True
                elif lose_flag:
                    self.show_overlay("Game Over")
                    self.game_over = True
                    self.simulating = False
                    self.simulate_button.config(text="Simulate Game")

    def TDL_agent_play(self):
        while self.simulating and not self.game_over:
            delay = self.delay_slider.get()
            time.sleep(delay / 1000.0)
            start = time.perf_counter()
            board = Board(self.board)
            
            best_move = tdl.select_best_move(board)
            
            if not best_move.is_valid():
                self.game_over = True
                self.show_overlay("Game Over")
                self.simulating = False
                self.simulate_button.config(text="Simulate Game")
                break
            
            opcode = best_move.opcode

            if opcode == 0:
                opcode = 0
            elif opcode == 1:
                opcode = 3
            elif opcode == 2:
                opcode = 1
            elif opcode == 3:
                opcode = 2
            
            new_board, score, moved, win_flag, lose_flag = play_game_gui_3x3(
                self.board, opcode, self.win_rank, already_won=self.win)
            
            move_duration = (time.perf_counter() - start) * 1000.0
            self.canvas.after(0, lambda md=move_duration: self.move_time_label.config(text=f"Move Time: {md:.1f} ms"))
            
            if moved:
                self.board = new_board
                self.score += int(score)
                if self.score > self.best_score:
                    self.best_score = self.score
                self.update_gui()
                
                if win_flag and not self.win:
                    self.win = True
                    self.show_overlay("You Win!")
                    self.game_over = True
                elif lose_flag:
                    self.show_overlay("Game Over")
                    self.game_over = True
                    self.simulating = False
                    self.simulate_button.config(text="Simulate Game")

    def new_game(self):
        self.board = initial_board_3x3()
        self.score = 0
        self.game_over = False
        self.win = False
        self.simulating = False
        self.last_move = None
        self.simulate_button.config(text="Simulate Game")
        self.clear_overlay()
        if self.suggestion_text:
            self.canvas.delete(self.suggestion_text)
            self.suggestion_text = None
        self.update_gui()
        self.canvas.focus_set()

    def update_gui(self):
        self.canvas.delete("tiles")
        self.canvas.create_rectangle(0, 0, BOARD_SIZE_3x3, BOARD_SIZE_3x3, fill=BOARD_COLOR, outline="", tags="tiles")
        for i in range(N_3x3):
            for j in range(N_3x3):
                x0 = CELL_PADDING + j * (CELL_SIZE + CELL_PADDING)
                y0 = CELL_PADDING + i * (CELL_SIZE + CELL_PADDING)
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                index = i * N_3x3 + j
                tile = (self.board >> (4 * index)) & 0xF
                value = 0 if tile == 0 else 1 << tile
                color = EMPTY_CELL_COLOR if value == 0 else TILE_COLORS.get(value, "#3c3a32")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="", tags="tiles")
                if value != 0:
                    text_color = TILE_TEXT_COLORS.get(value, "#f9f6f2")
                    font_size = 24 if value < 1024 else 20
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text=str(value),
                                            fill=text_color, font=("Helvetica", font_size, "bold"), tags="tiles")
        self.score_label.config(text=f"Score: {self.score}")
        self.best_label.config(text=f"Best: {self.best_score}")
        self.update_idletasks()
        if self.suggestion_visible and not self.game_over:
            self.update_suggestion()

    def key_handler(self, event):
        if self.game_over:
            return
        key = event.keysym
        if key == "Up":
            action = 0
        elif key == "Down":
            action = 1
        elif key == "Left":
            action = 2
        elif key == "Right":
            action = 3
        else:
            return

        new_board, score, moved, win_flag, lose_flag = play_game_gui_3x3(
            self.board, action, self.win_rank, already_won=self.win)
        if not moved:
            return
        self.board = new_board
        self.score += int(score)
        if self.score > self.best_score:
            self.best_score = self.score
        self.update_gui()
        if win_flag and not self.win:
            self.win = True
            self.show_overlay("You Win!")
            self.game_over = True
        elif lose_flag:
            self.show_overlay("Game Over")
            self.game_over = True

    def show_overlay(self, message):
        self.clear_overlay()
        if message == "Game Over":
            overlay = self.canvas.create_rectangle(0, 0, BOARD_SIZE_3x3, BOARD_SIZE_3x3,
                                                   fill="gray", stipple="gray50", outline="")
            text = self.canvas.create_text(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3//2 - 30, text=message,
                                           fill="white", font=("Helvetica", 32, "bold"))
            try_again_button = tk.Button(self.canvas, text="Try Again", font=("Helvetica", 16, "bold"),
                                         bg="#8f7a66", fg="white", command=self.new_game)
            button_window = self.canvas.create_window(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3//2 + 30, window=try_again_button)
            self.overlay_items.extend([overlay, text, button_window])
        elif message == "You Win!":
            overlay = self.canvas.create_rectangle(0, 0, BOARD_SIZE_3x3, BOARD_SIZE_3x3,
                                                   fill="orange", stipple="gray50", outline="")
            text = self.canvas.create_text(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3//2 - 50, text=message,
                                           fill="white", font=("Helvetica", 32, "bold"))
            keep_going_button = tk.Button(self.canvas, text="Keep Going", font=("Helvetica", 16, "bold"),
                                          bg="#8f7a66", fg="white", command=self.keep_going)
            try_again_button = tk.Button(self.canvas, text="Try Again", font=("Helvetica", 16, "bold"),
                                         bg="#8f7a66", fg="white", command=self.new_game)
            keep_going_window = self.canvas.create_window(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3//2 + 10, window=keep_going_button)
            try_again_window = self.canvas.create_window(BOARD_SIZE_3x3//2, BOARD_SIZE_3x3//2 + 60, window=try_again_button)
            self.overlay_items.extend([overlay, text, keep_going_window, try_again_window])

    def clear_overlay(self):
        for item in self.overlay_items:
            self.canvas.delete(item)
        self.overlay_items = []

    def keep_going(self):
        self.game_over = False
        self.clear_overlay()
        if self.simulating:
            agent = self.agent_var.get()
            if agent == "Random Agent":
                threading.Thread(target=self.random_agent_play, daemon=True).start()
            elif agent == "Value Iteration Agent":
                threading.Thread(target=self.value_iteration_agent, daemon=True).start()
            elif agent == "TDL Agent":
                threading.Thread(target=self.TDL_agent_play, daemon=True).start()

    def back_to_start(self):
        self.master.unbind_all("<Key>")
        self.destroy()
        starting_page = StartingPage(self.master)
        starting_page.pack(fill="both", expand=True)

def warmup_jit_functions():
    dummy_board = Board_4x4()
    dummy_board.raw = 0 
    _ = transpose_4x4(dummy_board.raw)
    _ = execute_move_left_4x4(dummy_board.raw, row_left_table_4x4)
    _ = execute_move_right_4x4(dummy_board.raw, row_right_table_4x4)
    _ = count_empty_4x4(dummy_board.raw)
    _ = score_board_4x4(dummy_board.raw)

class Game2048GUI_4x4(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BACKGROUND_COLOR)
        self.master = master
        self.master.title("2048")
        self.pack(fill="both", expand=True, padx=20, pady=20)

        self.simulating_4x4 = False
        self.score = 0.0
        self.best_score = 0.0
        self.game_over = False
        self.board = None
        self.overlay_items = []
        self.suggestion_text_4x4 = None

        self.performance_texts = {
            "TDL Agent": "64: 100.0%\n128: 100.0%\n256: 100.0%\n512: 100.0%\n1024: 99.9%\n2048: 86.8%\n4096: 49.9%\n8192: 3.8%"
        }

        self.controls_frame = tk.Frame(self, bg=BACKGROUND_COLOR)
        self.controls_frame.pack(pady=(0, 20), fill="x")

        self.header_frame = tk.Frame(self.controls_frame, bg=BACKGROUND_COLOR)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.header_frame.columnconfigure(0, weight=1)
        self.header_frame.columnconfigure(1, weight=1)
        self.header_frame.columnconfigure(2, weight=1)
        self.header_frame.columnconfigure(3, weight=1)

        self.score_label = tk.Label(self.header_frame, text="Score: 0", font=("Helvetica", 20, "bold"),
                                     bg=BACKGROUND_COLOR, fg="white", width=15, anchor="w")
        self.score_label.grid(row=0, column=0, sticky="w", padx=10)
        self.best_label = tk.Label(self.header_frame, text="Best: 0", font=("Helvetica", 20, "bold"),
                                    bg=BACKGROUND_COLOR, fg="white", width=15, anchor="w")
        self.best_label.grid(row=0, column=1, sticky="w", padx=10)
        self.new_game_button = tk.Button(self.header_frame, text="New Game", font=("Helvetica", 18, "bold"),
                                         bg="#8f7a66", fg="white", command=self.new_game)
        self.new_game_button.grid(row=0, column=2, sticky="e", padx=10)
        self.back_button = tk.Button(self.header_frame, text="Back", font=("Helvetica", 18, "bold"),
                                     bg="#8f7a66", fg="white", command=self.back_to_start)
        self.back_button.grid(row=0, column=3, sticky="e", padx=10)

        self.controls_buttons_frame_4x4 = tk.Frame(self.controls_frame, bg=BACKGROUND_COLOR)
        self.controls_buttons_frame_4x4.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.controls_buttons_frame_4x4.columnconfigure(0, weight=1)
        self.controls_buttons_frame_4x4.columnconfigure(1, weight=1)

        self.simulate_button_4x4 = tk.Button(self.controls_buttons_frame_4x4, text="Simulate Game",
                                             font=("Helvetica", 16, "bold"), bg="#8f7a66", fg="white",
                                             bd=0, padx=15, pady=10, activebackground="#9f8b76",
                                             command=self.toggle_simulation_4x4)
        self.simulate_button_4x4.grid(row=0, column=0, sticky="ew", padx=10)
        self.advice_button_4x4 = tk.Button(self.controls_buttons_frame_4x4, text="Show Move Advice",
                                           font=("Helvetica", 16, "bold"), bg="#8f7a66", fg="white",
                                           bd=0, padx=15, pady=10, activebackground="#9f8b76",
                                           command=self.toggle_suggestion_4x4)
        self.advice_button_4x4.grid(row=0, column=1, sticky="ew", padx=10)

        self.slider_frame_4x4 = tk.Frame(self.controls_frame, bg=BACKGROUND_COLOR)
        self.slider_frame_4x4.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.slider_frame_4x4.columnconfigure(0, weight=1)
        self.slider_frame_4x4.columnconfigure(1, weight=1)

        self.delay_slider_4x4 = tk.Scale(self.slider_frame_4x4, from_=0, to=500, orient="horizontal",
                                         label="Delay (ms)", resolution=1, length=250,
                                         bg=BACKGROUND_COLOR, fg="white", font=("Helvetica", 14),
                                         highlightthickness=0)
        self.delay_slider_4x4.set(10)
        self.delay_slider_4x4.grid(row=0, column=0, sticky="w", padx=10)
        self.move_time_label_4x4 = tk.Label(self.slider_frame_4x4, text="Move Time: 0.0 ms",
                                            font=("Helvetica", 16, "bold"), bg=BACKGROUND_COLOR, fg="white")
        self.move_time_label_4x4.grid(row=0, column=1, sticky="e", padx=10)

        self.board_frame = tk.Frame(self, bg=BACKGROUND_COLOR)
        self.board_frame.pack()
        self.canvas = tk.Canvas(self.board_frame, width=BOARD_SIZE_4x4, height=BOARD_SIZE_4x4+40,
                                bg=BOARD_COLOR, highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.performance_frame = tk.LabelFrame(self.board_frame, text="Performance", 
                                               font=("Helvetica", 14, "bold"),
                                               bg=BACKGROUND_COLOR, fg="white")
        self.performance_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        self.performance_text = tk.Text(self.performance_frame, width=20, height=10,
                                        bg=BACKGROUND_COLOR, fg="white", font=("Helvetica", 12),
                                        bd=0, highlightthickness=0)
        self.performance_text.pack(padx=10, pady=10)
        self.performance_text.config(state="disabled")

        self.master.bind_all("<Key>", self.key_handler)
        self.new_game()
        self.update_performance()

    def update_performance(self):
        perf_text = self.performance_texts.get("TDL Agent", "")
        self.performance_text.config(state="normal")
        self.performance_text.delete("1.0", tk.END)
        lines = perf_text.strip().split("\n")
        for i, line in enumerate(lines):
            parts = line.split(":")
            if len(parts) < 2:
                self.performance_text.insert(tk.END, line+"\n")
                continue
            perc_str = parts[1].strip().replace("%", "")
            try:
                perc = float(perc_str)
            except:
                perc = 0.0
            red = int(255 * (100 - perc) / 100)
            green = int(255 * perc / 100)
            color = f"#{red:02x}{green:02x}00"
            tag_name = f"line{i}"
            self.performance_text.insert(tk.END, line+"\n", tag_name)
            self.performance_text.tag_config(tag_name, foreground=color)
        self.performance_text.config(state="disabled")

    def toggle_simulation_4x4(self):
        if self.simulating_4x4:
            self.simulating_4x4 = False
            self.simulate_button_4x4.config(text="Simulate Game")
        else:
            self.simulating_4x4 = True
            self.simulate_button_4x4.config(text="Stop Simulation")
            threading.Thread(target=self.TDL_agent_play, daemon=True).start()

    def toggle_suggestion_4x4(self):
        if self.suggestion_text_4x4:
            self.canvas.delete(self.suggestion_text_4x4)
            self.suggestion_text_4x4 = None
            self.advice_button_4x4.config(text="Show Move Advice")
        else:
            best_move = tdl_4x4.select_best_move(self.board)
            if not best_move.is_valid():
                suggestion = "No moves available"
            else:
                directions = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
                suggestion = f"Suggested Move: {directions.get(best_move.opcode, 'Unknown')}"
            self.suggestion_text_4x4 = self.canvas.create_text(
                BOARD_SIZE_4x4//2, BOARD_SIZE_4x4 + 20,
                text=suggestion,
                fill="white",
                font=("Helvetica", 16, "bold")
            )
            self.advice_button_4x4.config(text="Hide Move Advice")

    def update_suggestion_4x4_sim(self):
        best_move = tdl_4x4.select_best_move(self.board)
        if not best_move.is_valid():
            suggestion = "No moves available"
        else:
            directions = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
            suggestion = f"Suggested Move: {directions.get(best_move.opcode, 'Unknown')}"
        if self.suggestion_text_4x4:
            self.canvas.itemconfigure(self.suggestion_text_4x4, text=suggestion)
        else:
            self.suggestion_text_4x4 = self.canvas.create_text(
                BOARD_SIZE_4x4 // 2,
                BOARD_SIZE_4x4 + 20,
                text=suggestion,
                fill="white",
                font=("Helvetica", 16, "bold")
            )

    def new_game(self):
        self.clear_overlay()
        self.board = Board_4x4()
        self.board.init()
        warmup_jit_functions()
        self.score = 0.0
        self.game_over = False
        self.update_gui()
        self.canvas.focus_set()

    def update_gui(self):
        self.canvas.delete("tiles")
        for i in range(N_4x4):
            for j in range(N_4x4):
                x0 = CELL_PADDING + j * (CELL_SIZE + CELL_PADDING)
                y0 = CELL_PADDING + i * (CELL_SIZE + CELL_PADDING)
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                index = i * N_4x4 + j
                tile = self.board.at(index)
                value = 0 if tile == 0 else 1 << tile
                color = EMPTY_CELL_COLOR if value == 0 else TILE_COLORS.get(value, "#3c3a32")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="", tags="tiles")
                if value != 0:
                    text_color = TILE_TEXT_COLORS.get(value, "#f9f6f2")
                    font_size = 24 if value < 1024 else 20
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text=str(value),
                                            fill=text_color, font=("Helvetica", font_size, "bold"), tags="tiles")
        self.score_label.config(text=f"Score: {self.score}")
        self.best_label.config(text=f"Best: {self.best_score}")
        self.update_idletasks()

        if self.suggestion_text_4x4:
            self.update_suggestion_4x4_sim()

    def key_handler(self, event):
        if self.game_over:
            return
        key = event.keysym
        if key == "Up":
            action = 0
        elif key == "Right":
            action = 1
        elif key == "Down":
            action = 2
        elif key == "Left":
            action = 3
        else:
            return

        move_score = self.board.move(action)
        if move_score != -1:
            self.score += move_score
            self.board.popup()
            self.update_gui()
            if not any(Board_4x4(self.board.raw).move(dir) != -1 for dir in [0, 1, 2, 3]):
                self.game_over = True
                self.show_overlay("Game Over")

    def show_overlay(self, message):
        overlay = self.canvas.create_rectangle(0, 0, BOARD_SIZE_4x4, BOARD_SIZE_4x4,
                                                 fill="gray", stipple="gray50", outline="")
        text = self.canvas.create_text(BOARD_SIZE_4x4//2, BOARD_SIZE_4x4//2 - 30, text=message,
                                        fill="white", font=("Helvetica", 32, "bold"))
        try_again_button = tk.Button(self.canvas, text="Try Again", font=("Helvetica", 16, "bold"),
                                     bg="#8f7a66", fg="white", command=self.new_game)
        button_window = self.canvas.create_window(BOARD_SIZE_4x4//2, BOARD_SIZE_4x4//2 + 30, window=try_again_button)
        self.overlay_items = [overlay, text, button_window]

    def clear_overlay(self):
        for item in self.overlay_items:
            self.canvas.delete(item)
        self.overlay_items = []

    def back_to_start(self):
        self.master.unbind_all("<Key>")
        self.destroy()
        starting_page = StartingPage(self.master)
        starting_page.pack(fill="both", expand=True)

    def TDL_agent_play(self):
        while self.simulating_4x4 and not self.game_over:
            delay = self.delay_slider_4x4.get()
            time.sleep(delay / 1000.0)
            start = time.perf_counter()

            if self.suggestion_text_4x4:
                self.canvas.after(0, self.update_suggestion_4x4_sim)
            
            best_move = tdl_4x4.select_best_move(self.board)
            if not best_move.is_valid():
                self.game_over = True
                self.canvas.after(0, lambda: self.show_overlay("Game Over"))
                self.simulating_4x4 = False
                self.simulate_button_4x4.config(text="Simulate Game")
                break
            
            move_score = self.board.move(best_move.opcode)
            if move_score != -1:
                self.score += move_score
                self.board.popup()
                self.canvas.after(0, self.update_gui)
            
            move_duration = (time.perf_counter() - start) * 1000.0
            self.canvas.after(0, lambda md=move_duration: self.move_time_label_4x4.config(text=f"Move Time: {md:.1f} ms"))
            
            if not any(Board_4x4(self.board.raw).move(dir) != -1 for dir in [0, 1, 2, 3]):
                self.game_over = True
                self.canvas.after(0, lambda: self.show_overlay("Game Over"))
                self.simulating_4x4 = False
                self.simulate_button_4x4.config(text="Simulate Game")
                break