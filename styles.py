#styles.py
from tkinter import ttk

def apply_style(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("TNotebook", background="#111318", borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 11, "bold"))
    style.map("TNotebook.Tab",background=[("selected", "#1c2028"), ("!selected", "#12151b")],foreground=[("selected", "#ffffff"), ("!selected", "#cbd5e1")])

    style.configure("Card.TFrame", background="#171a21", relief="flat")
    style.configure("Title.TLabel", background="#171a21", foreground="#ffffff", font=("Segoe UI", 14, "bold"))
    style.configure("Body.TLabel", background="#171a21", foreground="#cbd5e1", font=("Segoe UI", 10))
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 8))
    style.configure("TLabel", font=("Segoe UI", 10))
