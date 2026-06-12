"""Entry point for `python -m airsketch.classroom`."""
import os

from airsketch.classroom.app import main

if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
