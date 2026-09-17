import time
import json
import argparse
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from vishustra_core.engine import VishustraEngine
from vishustra_core.kernel import VishustraKernel, VishustraKernelError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KERNEL: VishustraKernel = VishustraKernel()


def run_dashboard() -> None:
    logger.info("Initializing Vishustra AI Orchestrator Dashboard")

    engine = VishustraEngine()
    active_nodes = engine.get_active_nodes()

    logger.info("System Health: ONLINE")
    logger.info(f"Total Active Modules/Nodes: {len(active_nodes)}")

    if active_nodes:
        logger.info("Loaded Modules:")
        for idx, name in enumerate(active_nodes[-10:], 1):
            logger.info(f"  {idx}. {name}")
        if len(active_nodes) > 10:
            logger.info(f"  ... and {len(active_nodes) - 10} more.")

        engine.run_simulation("Test input packet from main dashboard.")
    else:
        logger.warning("No active nodes found. Please configure the node pipeline to continue.")

    show_kernel_card()


def show_kernel_card() -> None:
    from vishustra_core.kernel.skills import SKILLS
    count = len(SKILLS)
    logger.info("-" * 62)
    logger.info(f"VISHUSTRA KERNEL online | {count} composable skills registered")
    logger.info("Try: python -m vishustra_core.kernel.demo --scenario review")
    logger.info("Or:  python -m vishustra_core.main --kernel  \"<intention>\" \"<some text>\"")
    logger.info("Or:  python -m vishustra_core.main --serve   (REST API on port 8812)")
    logger.info("-" * 62)


def run_kernel_cli(intent: str, text: str, raw: bool) -> int:
    try:
        report = KERNEL.process(intent, text)
    except VishustraKernelError as e:
        print(f"KERNEL ERROR: {e}")
        return 1
    if raw:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\nINTENT  : {report['intent']}")
        print(f"MODE    : {report['mode']}")
        print(f"PIPELINE: {' -> '.join(report['pipeline'])}")
        print(f"FINAL   : {json.dumps(report['result']['final'], ensure_ascii=False, default=str)}")
    return 0


class KernelHTTPHandler(BaseHTTPRequestHandler):
    """Tiny zero-dependency REST surface: POST /kernel, GET /health."""

    def log_message(self, fmt, *args):  # keep default server logging quiet
        logger.info(f"[kernel-api] {self.address_string()} - {fmt % args}")

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/health"):
            self._json(200, {"status": "online", "health": "OK", "service": "vishustra-kernel"})
        else:
            self._json(404, {"error": "not found", "hint": "POST /kernel with {'intent': ..., 'text': ...}"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/kernel"):
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            intent = body.get("intent", "")
            text = body.get("text", "")
            if not intent:
                self._json(400, {"error": "missing 'intent'"})
                return
            report = KERNEL.process(intent, text)
            self._json(200, report)
        except VishustraKernelError as e:
            self._json(422, {"error": str(e)})
        except Exception as e:
            self._json(500, {"error": str(e)})


def run_kernel_server(port: int = 8812) -> None:
    show_kernel_card()
    server = ThreadingHTTPServer(("0.0.0.0", port), KernelHTTPHandler)
    logger.info(f"VISHUSTRA KERNEL REST API listening on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Kernel REST API shutting down.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Vishustra orchestrator")
    parser.add_argument("--serve", action="store_true", help="expose the Kernel as a REST API")
    parser.add_argument("--port", type=int, default=8812, help="port for --serve")
    parser.add_argument("--kernel", nargs=2, metavar=("INTENT", "TEXT"), help="run one Kernel pipeline directly")
    parser.add_argument("--raw", action="store_true", help="print raw JSON with --kernel")
    args = parser.parse_args(argv)

    if args.serve:
        run_kernel_server(args.port)
        return 0
    if args.kernel:
        return run_kernel_cli(*args.kernel, raw=args.raw)
    run_dashboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
