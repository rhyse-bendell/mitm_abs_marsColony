from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict


def fill_text_widget(widget: tk.Text, payload: Any):
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    if isinstance(payload, str):
        widget.insert(tk.END, payload)
    else:
        widget.insert(tk.END, json.dumps(payload, indent=2, default=str))
    widget.configure(state="disabled")


def populate_key_value_tree(tree: ttk.Treeview, data: Dict[str, Any]):
    for item in tree.get_children():
        tree.delete(item)
    for k, v in data.items():
        tree.insert("", "end", values=(k, v))
