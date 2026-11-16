"""Owned loopback-only Showdown subprocess with bounded startup and cleanup."""
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]

@contextmanager
def local_server(port, output):
    checkout = ROOT / 'vendor/pokemon-showdown'
    revision = (ROOT / 'configs/showdown-revision.txt').read_text().strip()
    actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != actual:
        raise RuntimeError('Showdown revision differs from configured pin')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} already occupied; refusing to reuse unknown server')
    config = checkout / 'config/config.js'
    config.write_text("exports.port = %d;\nexports.bindaddress = '127.0.0.1';\nexports.noguestsecurity = true;\nexports.nothrottle = true;\nexports.workers = 1;\nexports.simulatorprocesses = 1;\nexports.validatorprocesses = 1;\nexports.reportjoins = false;\n" % port)
    with open(output, 'w') as log:
        # Own the process group, including simulator/validator workers.
        process = subprocess.Popen(['node', 'pokemon-showdown', 'start', str(port)],
                                   cwd=checkout, stdout=log, stderr=log, start_new_session=True)
        try:
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError(f'Showdown exited; see {output}')
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', port)) == 0:
                        break
                time.sleep(.1)
            else:
                raise TimeoutError('Showdown startup exceeded 15 seconds')
            yield
        finally:
            import os, signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
