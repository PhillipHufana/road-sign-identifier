# eco_facts.py
import tkinter as tk

def show_eco_fact(parent, fact_text: str, headline: str):
    win = tk.Toplevel(parent)
    win.title("Eco Fact Card 🌳")
    win.resizable(False, False)

    frm = tk.Frame(win, padx=14, pady=12)
    frm.pack(fill="both", expand=True)

    tk.Label(frm, text="Forest Edition", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(frm, text=headline, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 10))
    tk.Label(frm, text=f"“{fact_text}”", font=("Segoe UI", 11), wraplength=460, justify="left").pack(anchor="w")
    tk.Button(frm, text="Close", width=12, command=win.destroy).pack(anchor="e", pady=(12, 0))
