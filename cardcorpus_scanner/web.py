"""The local web page: a photo in (upload, paste or webcam), the cards out.

`cardcorpus-scan serve` runs it on http://127.0.0.1:8000. The page sends the
photo here, the scanner names the cards, and the page shows each with what
the CardCorpus dataset says about it, read from cardcorpus.com/v1 in the
browser.

POST /api/scan answers in lines of JSON: first the cards by picture alone
(`"first": true`, as soon as the cards are fingerprinted), then the final
answer once the later stages have double-checked them, or {"error": ...}.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
MAX_BYTES = 40 * 1024 * 1024


def make_server(scanner, host="127.0.0.1", port=8000):
    """The page's HTTP server, not yet serving (port 0 picks a free one)."""
    lock = threading.Lock()  # one scan at a time: the models are shared
    info = {**scanner.index.release, "verifier": scanner.verifier is not None}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send(200, info)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/api/scan":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n <= 0 or n > MAX_BYTES:
                return self._send(413, {"error": "send one photo, under 40 MB"})
            body = self.rfile.read(n)
            families = [f for f in (self.path.split("language=")[1:] or [""])[0].split(",") if f] or None
            try:
                from PIL import Image

                photo = Image.open(BytesIO(body))
                photo.load()
            except Exception:
                return self._send(400, {"error": "that isn't a photo this can read"})
            # lines of JSON, each written as soon as it is known; the
            # connection closes after the last (HTTP/1.0, no length)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def line(body):
                self.wfile.write(json.dumps(body, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()

            with lock:
                try:
                    line(scanner.scan(photo, families=families, on_first=line, on_progress=line))
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the page went away
                except Exception as e:  # report it on the page rather than drop the connection
                    line({"error": f"the scan failed: {e}"})

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(scanner, host="127.0.0.1", port=8000):
    server = make_server(scanner, host, port)
    print(f"open http://{host}:{server.server_address[1]} (Ctrl-C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
