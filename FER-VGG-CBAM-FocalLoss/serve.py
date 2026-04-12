"""
Run this script from the same folder as index.html.
It starts a local HTTP server so the browser treats
the page as http:// (not file://) — required for fetch() to work.

Usage:
  1. Open a terminal / command prompt
  2. cd into the folder containing index.html and this file
  3. Run:  python serve.py
  4. Open your browser at:  http://localhost:8000
"""

import http.server, socketserver, webbrowser, os

PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))

handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), handler) as httpd:
    url = f"http://localhost:{PORT}"
    print(f"\n  Server running at  {url}")
    print(f"  Serving files from {os.getcwd()}")
    print("\n  Open your browser at http://localhost:8000")
    print("  Press Ctrl+C to stop.\n")
    webbrowser.open(url)
    httpd.serve_forever()
