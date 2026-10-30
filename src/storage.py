"""Small SQLite record store, using Python's sqlite3 tutorial transaction pattern.

Reference: https://docs.python.org/3/library/sqlite3.html#tutorial
Models are lossless NPZ BLOBs. Inventory describes simulated deployment payload,
not a clinical stock ledger. Each write uses parameters and an atomic transaction.
"""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import uuid

DB_PATH = Path(__file__).resolve().parents[1] / 'outputs' / 'microdose.sqlite3'
SCHEMA = '''
CREATE TABLE IF NOT EXISTS models (
    key TEXT PRIMARY KEY, metadata TEXT NOT NULL, checkpoint BLOB NOT NULL,
    sha256 TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY, instruction TEXT NOT NULL, layout INTEGER NOT NULL,
    mode TEXT NOT NULL, status TEXT NOT NULL, timestep INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tasks (
    deployment_id TEXT NOT NULL REFERENCES deployments(id), robot TEXT NOT NULL,
    sequence_index INTEGER NOT NULL, target TEXT NOT NULL, dose REAL NOT NULL,
    controller TEXT NOT NULL, status TEXT NOT NULL, consumed REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (deployment_id, robot, sequence_index)
);
CREATE TABLE IF NOT EXISTS inventory (
    deployment_id TEXT NOT NULL REFERENCES deployments(id), robot TEXT NOT NULL,
    reagents TEXT NOT NULL, capacity REAL NOT NULL, allocated REAL NOT NULL,
    consumed REAL NOT NULL DEFAULT 0, remaining REAL NOT NULL,
    PRIMARY KEY (deployment_id, robot),
    CHECK (allocated >= 0 AND allocated <= capacity),
    CHECK (consumed >= 0 AND remaining >= 0)
);
CREATE TABLE IF NOT EXISTS robot_status (
    deployment_id TEXT NOT NULL REFERENCES deployments(id), robot TEXT NOT NULL,
    x INTEGER NOT NULL, y INTEGER NOT NULL, status TEXT NOT NULL,
    timestep INTEGER NOT NULL, PRIMARY KEY (deployment_id, robot)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT NOT NULL,
    details TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
'''


@contextmanager
def connection(path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA foreign_keys = ON')
        db.execute('PRAGMA journal_mode = WAL')
        db.executescript(SCHEMA)
        with db:
            yield db
    finally:
        db.close()


def save_model(key, metadata, checkpoint, path=DB_PATH):
    checksum = sha256(checkpoint).hexdigest()
    with connection(path) as db:
        db.execute('''INSERT INTO models(key, metadata, checkpoint, sha256)
            VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET
            metadata=excluded.metadata, checkpoint=excluded.checkpoint,
            sha256=excluded.sha256, updated_at=strftime('%Y-%m-%d %H:%M:%f','now')''',
            (key, json.dumps(metadata), checkpoint, checksum))


def import_checkpoints(directory, path=DB_PATH):
    # Import missing legacy exports. existing SQL records remain authoritative.
    directory = Path(directory)
    manifest = directory / 'manifest.json'
    if not manifest.exists():
        return 0
    entries = json.loads(manifest.read_text())
    imported = 0
    with connection(path) as db:
        for key, metadata in entries.items():
            if metadata.get('objective') != 'safety':
                continue
            if db.execute('SELECT 1 FROM models WHERE key=?', (key,)).fetchone():
                continue
            data = (directory / metadata['checkpoint']).read_bytes()
            db.execute('INSERT INTO models(key,metadata,checkpoint,sha256) VALUES(?,?,?,?)',
                       (key, json.dumps(metadata), data, sha256(data).hexdigest()))
            imported += 1
    return imported


def read_model(key, path=DB_PATH):
    with connection(path) as db:
        row = db.execute('SELECT metadata,checkpoint,sha256 FROM models WHERE key=?', (key,)).fetchone()
    if row is None:
        raise FileNotFoundError(f'No saved SQL policy: {key}. Prepare policies offline first.')
    if sha256(row['checkpoint']).hexdigest() != row['sha256']:
        raise ValueError('Stored checkpoint checksum does not match.')
    return json.loads(row['metadata']), row['checkpoint']


def model_revision(path=DB_PATH):
    with connection(path) as db:
        return tuple(db.execute('SELECT count(*), max(updated_at) FROM models').fetchone())


def record_event(session_id, kind, details, path=DB_PATH):
    with connection(path) as db:
        db.execute('INSERT INTO events(session_id,kind,details) VALUES(?,?,?)',
                   (session_id, kind, json.dumps(details)))


def start_deployment(fleet, missions, instruction, path=DB_PATH):
    from src.verification.smt_verifier import RULES
    run_id = uuid.uuid4().hex
    with connection(path) as db:
        db.execute('INSERT INTO deployments(id,instruction,layout,mode,status) VALUES(?,?,?,?,?)',
                   (run_id, instruction, fleet['seed'], fleet['mode'], 'running'))
        for mission in missions:
            db.execute('''INSERT INTO tasks(deployment_id,robot,sequence_index,target,dose,controller,status)
                VALUES(?,?,?,?,?,?,?)''', (run_id, mission.carrier_id, mission.sequence_index,
                mission.target_id, mission.dose, mission.controller, 'waiting'))
        for robot in fleet['carriers']:
            allocated = sum(m.dose for m in missions if m.carrier_id == robot)
            db.execute('''INSERT INTO inventory(deployment_id,robot,reagents,capacity,allocated,remaining)
                VALUES(?,?,?,?,?,?)''', (run_id, robot, ' + '.join(RULES['carrier_reagents'][robot]),
                RULES['carrier_capacities'][robot], allocated, allocated))
    fleet['database_id'] = run_id
    sync_deployment(fleet, path)
    return run_id


def sync_deployment(fleet, path=DB_PATH):
    # Replace current status atomically. consumed amounts are totals, never deltas.
    run_id = fleet.get('database_id')
    if not run_id:
        return
    status = 'needs_revision' if fleet.get('needs_react') else 'completed' if fleet['done'] else 'running'
    with connection(path) as db:
        db.execute('UPDATE deployments SET status=?,timestep=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                   (status, fleet['tick'], run_id))
        # An absent task after a failed prior delivery is skipped, not still waiting.
        db.execute("UPDATE tasks SET status='skipped' WHERE deployment_id=? AND status IN ('waiting','running')", (run_id,))
        completed = {(t['robot'], t['sequence_index']): t for t in fleet['completed_tasks']}
        for robot, state in fleet['carriers'].items():
            for mission in fleet['queues'].get(robot, []):
                db.execute("UPDATE tasks SET status='waiting' WHERE deployment_id=? AND robot=? AND sequence_index=?",
                           (run_id, robot, mission.sequence_index))
            completed[(robot, state['sequence_index'])] = dict(terminal=state['terminal'], dose=state['dose'])
            db.execute('''INSERT INTO robot_status VALUES(?,?,?,?,?,?)
                ON CONFLICT(deployment_id,robot) DO UPDATE SET x=excluded.x,y=excluded.y,
                status=excluded.status,timestep=excluded.timestep''',
                (run_id, robot, *state['actual'], state['terminal'] or 'running', fleet['tick']))
        for (robot, index), task in completed.items():
            terminal = task['terminal'] or 'running'
            consumed = task['dose'] if terminal in ('delivery', 'off_target') else 0
            db.execute('UPDATE tasks SET status=?,consumed=? WHERE deployment_id=? AND robot=? AND sequence_index=?',
                       (terminal, consumed, run_id, robot, index))
        db.execute('''UPDATE inventory SET consumed=(SELECT COALESCE(sum(t.consumed),0)
            FROM tasks t WHERE t.deployment_id=inventory.deployment_id AND t.robot=inventory.robot)
            WHERE deployment_id=?''', (run_id,))
        db.execute('UPDATE inventory SET remaining=allocated-consumed WHERE deployment_id=?', (run_id,))


def cancel_deployment(fleet, path=DB_PATH):
    if not fleet or not fleet.get('database_id'):
        return
    sync_deployment(fleet, path)
    with connection(path) as db:
        run_id = fleet['database_id']
        db.execute("UPDATE deployments SET status='cancelled',updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='running'", (run_id,))
        db.execute("UPDATE tasks SET status='cancelled' WHERE deployment_id=? AND status IN ('running','waiting')", (run_id,))
        db.execute("UPDATE robot_status SET status='cancelled' WHERE deployment_id=? AND status='running'", (run_id,))


def recent_records(path=DB_PATH):
    with connection(path) as db:
        deployments = [dict(r) for r in db.execute('SELECT id,layout,status,timestep,created_at,updated_at FROM deployments ORDER BY rowid DESC LIMIT 10')]
        tasks = [dict(r) for r in db.execute('''SELECT t.* FROM tasks t JOIN deployments d ON d.id=t.deployment_id
            ORDER BY d.rowid DESC,t.sequence_index,t.robot LIMIT 30''')]
        inventory = [dict(r) for r in db.execute('''SELECT i.* FROM inventory i JOIN deployments d ON d.id=i.deployment_id
            ORDER BY d.rowid DESC,i.robot LIMIT 30''')]
    return deployments, tasks, inventory


if __name__ == '__main__':
    from src.algorithms.deployment import POLICY_DIR
    print(f'Imported {import_checkpoints(POLICY_DIR)} checkpoints into {DB_PATH}')
