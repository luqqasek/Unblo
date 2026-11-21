"""Build and serve only curated dashboard files on loopback. Ctrl+C stops."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from build import build, DIST

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--port',type=int,default=8877); args=ap.parse_args()
    build()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),functools.partial(SimpleHTTPRequestHandler,directory=str(DIST)))
    print(f'Dashboard: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
