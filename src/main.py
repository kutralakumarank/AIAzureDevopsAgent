"""Streamlit entry point for the test engineering platform."""
from dotenv import load_dotenv

from src.ui.streamlitui.test_engineering import render


def load_test_engineering_app():
    load_dotenv()
    render()
