from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import itertools
import json
import threading

RELIABLE_MOCK_RESPONSE_PATH = Path(__file__).parent / "mock-response-reliable-paper.json"
UNRELIABLE_MOCK_RESPONSE_PATH = Path(__file__).parent / "mock-response-unreliable-paper.json"

_response_cycle_lock = threading.Lock()
_response_paths = itertools.cycle([RELIABLE_MOCK_RESPONSE_PATH, UNRELIABLE_MOCK_RESPONSE_PATH])


def _next_mock_response_path() -> Path:
    """Alternate reliable/unreliable/reliable/... across successive /paper/search requests."""
    with _response_cycle_lock:
        return next(_response_paths)


class MockResponseHandler(BaseHTTPRequestHandler):
    
    def _set_headers(self, status_code=200):
        """Helper to set common JSON response headers."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        # Optional: Add CORS headers if you are calling this from a web browser
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests based on the URL path."""
        # print(f"GET {self.path} from {self.client_address[0]}")
        if self.path.startswith("/paper/search"):
            self._set_headers(200)
            with open(_next_mock_response_path(), "r", encoding="utf-8") as f:
                mock_data = json.load(f)
            self.wfile.write(json.dumps(mock_data).encode("utf-8"))
        else:
            # Return a 404 for unknown paths
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MockResponseHandler)
    print(f"Mock server running on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping mock server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server(port=8000)