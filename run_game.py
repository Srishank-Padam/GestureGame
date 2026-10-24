"""
run_game.py

Unified launcher for the Gesture Game project.
1. Starts a lightweight local HTTP server serving the Cyber Surfer runner game.
2. Automatically opens the game in your default web browser.
3. Launches the real-time AI gesture controller with live webcam feed and HUD.

Usage:
    python run_game.py
"""

import os
import sys
import time
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, HTTPServer
import functools

# Find an available port
def find_free_port(start_port=8000):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 8000


def start_http_server(directory, port):
    handler = functools.partial(SimpleHTTPRequestHandler, directory=directory)
    # Suppress verbose request logging to keep console clean
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    port = find_free_port(8000)

    print("\n" + "=" * 65)
    print("      CYBER SURFER : GESTURE-CONTROLLED RUNNER")
    print("=" * 65)
    print(f"[*] Starting local game server on http://localhost:{port}/game.html ...")
    start_http_server(base_dir, port)

    game_url = f"http://localhost:{port}/game.html"
    print(f"[*] Opening game in default browser: {game_url}")
    webbrowser.open(game_url)

    print("\n[*] Initializing CNN model and webcam...")
    print("    - Focus the browser window to play the game!")
    print("    - Keep the webcam window visible so you can monitor gestures.")
    print("    - Press SPACE in the webcam window to START/PAUSE sending keys.")
    print("=" * 65 + "\n")

    # Import and run game_control
    import game_control
    game_control.main()


if __name__ == "__main__":
    main()
