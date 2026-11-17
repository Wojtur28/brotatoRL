"""Web panel for choosing the parameters of a model/bot run.

    python python\\scripts\\webapp.py                 # http://127.0.0.1:8765
    python python\\scripts\\webapp.py --open          # and open the browser
    python python\\scripts\\webapp.py --run python\\presets\\moj.json   # same run without the browser

The panel listens on 127.0.0.1 only (no authentication).
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path

from brotato_rl.webui.runner import run_config
from brotato_rl.webui.server import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the panel in the default browser")
    parser.add_argument("--run", type=Path, help="run a saved preset (JSON) in the console and exit")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Polish log lines on a cp1252 console
        stream.reconfigure(errors="replace")

    if args.run:
        status = run_config(json.loads(args.run.read_text(encoding="utf-8")))
        print(json.dumps(status.get("summary"), indent=1))
        sys.exit(0 if status["state"] == "done" else 1)

    if args.port < 1 or args.port > 65535:
        parser.error("bad port")
    httpd, app = serve(port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"BrotatoRL panel: {url}   (Ctrl+C zatrzymuje)")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.runner.stop()
        if app.runner.thread:
            app.runner.thread.join(timeout=15)
        httpd.server_close()


if __name__ == "__main__":
    main()
