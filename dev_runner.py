import subprocess
import sys
import time
from pathlib import Path


WATCH_EXTENSIONS = {".py", ".env", ".example", ".txt", ".md"}
POLL_INTERVAL_SECONDS = 1.0


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", "generated_audio", "conversations", "logs"} for part in path.parts):
            continue
        if path.suffix.lower() in WATCH_EXTENSIONS or path.name == ".env":
            yield path


def snapshot(root: Path) -> dict[str, int]:
    state = {}
    for path in iter_files(root):
        try:
            state[str(path)] = path.stat().st_mtime_ns
        except OSError:
            continue
    return state


def main() -> int:
    root = Path(__file__).resolve().parent
    command = [sys.executable, "bot.py"]
    previous = snapshot(root)
    process = subprocess.Popen(command, cwd=root)
    print("Dev runner started. Watching for file changes...")

    try:
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            current = snapshot(root)

            if current != previous:
                previous = current
                print("Change detected. Restarting bot...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                process = subprocess.Popen(command, cwd=root)

            if process.poll() is not None:
                print(f"Bot exited with code {process.returncode}. Restarting in 2 seconds...")
                time.sleep(2)
                process = subprocess.Popen(command, cwd=root)
    except KeyboardInterrupt:
        print("Stopping dev runner...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
