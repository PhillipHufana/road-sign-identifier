import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.core.qmimedatabase=false;qt.qpa.*=false")

import tkinter as tk
from app import App

def main():
    root = tk.Tk()
    _ = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
