"""Chạy giao diện: python run_gui.py  (tương đương python -m gui_app)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_app.main import main

main()
