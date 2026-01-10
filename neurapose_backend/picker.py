import tkinter as tk
from tkinter import filedialog
import sys

def pick_folder():
    try:
        root = tk.Tk()
        root.withdraw()
        # Set to topmost so it doesn't appear behind the browser
        root.attributes("-topmost", True)
        
        # Open dialog
        folder = filedialog.askdirectory()
        
        # Cleanup
        root.destroy()
        
        if folder:
            # Print only the path so it can be captured by the parent process
            print(folder)
            sys.exit(0)
        else:
            # Cancelled or no folder selected
            sys.exit(1)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    pick_folder()
