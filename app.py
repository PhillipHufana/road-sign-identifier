# app.py
import tkinter as tk
from tkinter import ttk

from config import APP_TITLE, WINDOW_SIZE
from styles import apply_style
from game_tab import GameTab
from evidence_tab import EvidenceTab

def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry(WINDOW_SIZE)

    apply_style(root)

    # Main container
    container = ttk.Frame(root)
    container.pack(fill="both", expand=True)

    nb = ttk.Notebook(container)
    nb.pack(fill="both", expand=True, padx=10, pady=10)

    tab_game = GameTab(nb)
    tab_evidence = EvidenceTab(nb)

    nb.add(tab_game, text="🎮 Game Mode")
    nb.add(tab_evidence, text="🧰 Community Mode")

    root.mainloop()

if __name__ == "__main__":
    main()
