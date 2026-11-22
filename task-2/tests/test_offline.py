"""Offline check of the query, format and metric code against a local fake of the hackathon server.

Sends nothing to the real server and spends no budget. Run after editing src/meq:
    MEQ_API_KEY=test-key uv run python tests/test_offline.py
"""
import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from meq import api, config, scoring

LIMIT = {"ref-01": 10**6, "ref-02": 10**6, "ep-01": 100, "ep-02": 100}
used = {"ref-01": 0, "ref-02": 0, "ep-01": 97, "ep-02": 0}
posts, lock = [0], threading.Lock()


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, headers=()):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        assert self.headers["X-API-Key"] == "test-key"
        self.reply(200, {"data": [{"id": ep, "budget": {"limit": LIMIT[ep], "used": used[ep]}} for ep in LIMIT]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        ep = req["model"]
        with lock:
            posts[0] += 1
            n = posts[0]
            if n % 5 == 0:
                return self.reply(429, {"detail": "busy"}, [("Retry-After", "0.05")])
            if n == 7:
                return self.reply(503, {"detail": "backend down."})
            if used[ep] >= LIMIT[ep]:
                return self.reply(429, {"detail": f"{ep}: budget used up"})
            used[ep] += 1
            body = {"choices": [{"message": {"content": f"{ep} says: {req['messages'][0]['content']}"}}],
                    "budget": {"limit": LIMIT[ep], "used": used[ep]}, "usage": {"completion_tokens": 7}}
        self.reply(200, body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config.BASE_URL = f"http://127.0.0.1:{server.server_port}/{config.TASK_ID}/v1"

    tmp = Path(tempfile.mkdtemp())
    out = tmp / "answers.jsonl"
    prompts = [(f"p{i}", f"prompt number {i}") for i in range(4)]
    eps = ["ep-01", "ep-02"]

    print("--- 1. dry run plans, sends nothing")
    assert api.query(prompts, eps, out, dry_run=True) == 7        # ep-01 has only 3 left
    assert posts[0] == 0 and not out.exists()

    print("--- 2. real run survives 429-busy and 503, stays inside budget")
    assert api.query(prompts, eps, out) == 7
    rows = api.load_answers(out)
    assert len(rows) == 7 and used == {"ref-01": 0, "ref-02": 0, "ep-01": 100, "ep-02": 4}, (len(rows), used)
    assert len({(r["endpoint"], r["id"], r["repeat"]) for r in rows}) == 7
    assert all(r["response"].startswith(r["endpoint"]) and r["raw"]["usage"] for r in rows)

    print("--- 3. rerun sends nothing new")
    before = posts[0]
    assert api.query(prompts, eps, out) == 0 and posts[0] == before

    print("--- 4. an id reused for another prompt is refused")
    try:
        api.query([("p0", "something else")], eps, out)
        raise AssertionError("clash not caught")
    except SystemExit as exc:
        print("   refused:", exc)

    print("--- 5. budget running out mid-run stops that endpoint only")
    used["ep-02"] = 98
    out2 = tmp / "more.jsonl"
    used_before = dict(used)
    api.query([(f"q{i}", f"q {i}") for i in range(2)], eps, out2)   # planned from server budget: 2 on ep-02, 0 on ep-01
    assert used["ep-02"] == 100 and used["ep-01"] == used_before["ep-01"]
    assert len(api.load_answers(out2)) == 2

    print("--- 6. reference run with repeats; text prompts get stable hashed ids")
    txt = tmp / "p.txt"
    txt.write_text("first prompt\n\nsecond prompt\n")
    ids = [pid for pid, _ in api.read_prompts(txt)]
    txt.write_text("a new first line\nfirst prompt\nsecond prompt\n")
    assert [pid for pid, _ in api.read_prompts(txt)][1:] == ids
    out3 = tmp / "ref.jsonl"
    assert api.query(api.read_prompts(txt), ["ref-01", "ref-02"], out3, repeats=2) == 12
    assert len(api.load_answers(out3)) == 12

    print("--- 7. unknown endpoint is refused before anything is sent")
    try:
        api.query(prompts, ["ep-99"], out)
        raise AssertionError("unknown endpoint not caught")
    except SystemExit as exc:
        print("   refused:", exc)

    print("--- 8. a model label shared by two families is flagged")
    sub = tmp / "sub.csv"
    sub.write_text("".join(f"ep-{i:02d},{'A' if i < 20 else 'B'},shared\n" for i in range(1, 34)))
    problems = scoring.validate(sub)[1]
    assert len(problems) == 1 and "several families" in problems[0], problems

    print("--- 9. full metric: perfect answer scores 1, families only scores 0.1")
    truth = {f"ep-{i:02d}": (f"F{i % 3}", f"M{i % 6}", "") for i in range(1, 34)}
    assert abs(scoring.score(truth, truth)["score"] - 1.0) < 1e-9
    fam_only = {ep: (f, ep, "") for ep, (f, _, _) in truth.items()}
    assert abs(scoring.score(truth, fam_only)["score"] - 0.1) < 1e-9

    shutil.rmtree(tmp)
    print("ALL OK")


if __name__ == "__main__":
    main()
