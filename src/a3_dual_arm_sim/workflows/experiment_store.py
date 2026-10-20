"""Single-writer, crash-resumable experiment records (Linux local filesystem)."""
import fcntl
import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ExperimentStore:
    def __init__(self, root, protocol):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'writer.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError('Experiment already has an active writer') from None
        try:
            path = self.root / 'protocol.json'
            if path.exists():
                if json.loads(path.read_text()) != protocol:
                    raise RuntimeError('Protocol changed; use a new experiment directory')
            else:
                atomic_json(path, protocol)
            (self.root / 'records').mkdir(exist_ok=True)
            self.records = {}
            for path in (self.root / 'records').glob('*.json'):
                row = json.loads(path.read_text())
                key = row['key']
                if key in self.records or path.stem != key:
                    raise RuntimeError('Duplicate or mismatched episode key')
                self.records[key] = row
        except BaseException:
            self.close()
            raise

    def commit(self, key, row):
        if key in self.records:
            raise RuntimeError(f'Duplicate episode: {key}')
        value = dict(row, key=key)
        atomic_json(self.root / 'records' / f'{key}.json', value)
        self.records[key] = value

    def close(self):
        self.lock.close()
