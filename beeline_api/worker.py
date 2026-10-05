"""GPU worker: runs one job at a time (extraction or solve).

    python -m beeline_api.worker

Jobs are claimed with SELECT ... FOR UPDATE SKIP LOCKED. A job whose heartbeat stopped
(worker killed) is marked failed on the next start, which frees the single-job slot.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
import traceback

from . import db
from .extraction import run_extraction
from .jobs import Cancelled, Reporter
from .solvejob import run_solve_job

log = logging.getLogger("beeline.worker")
STALE_S = 180
_stop = threading.Event()


def recover_stale(startup: bool = False):
    """There is one worker. At startup nothing can be running, so any 'running' job is orphaned."""
    age = 0 if startup else STALE_S
    rows = db.all_("""update jobs set status='failed', finished_at=now(),
                         error='worker stopped while this job was running; start it again'
                      where status='running' and coalesce(heartbeat_at, started_at, created_at) <= now() - (%s || ' seconds')::interval
                      returning id, run_id, kind""", (age,))
    for r in rows:
        db.exec_("""update runs set status = case when %s='extract' then 'review' else
                        case when latest_solve_id is null then 'review' else 'complete' end end
                    where id=%s""", (r["kind"], r["run_id"]))
        log.warning("recovered stale job %s", r["id"])


def claim() -> dict | None:
    with db.conn() as c:
        row = c.execute("""select * from jobs where status='queued' order by id limit 1 for update skip locked""").fetchone()
        if not row:
            return None
        if row["cancel_requested"]:
            c.execute("update jobs set status='cancelled', finished_at=now() where id=%s", (row["id"],))
            return None
        c.execute("update jobs set status='running', started_at=now(), heartbeat_at=now(), stage='Starting' where id=%s",
                  (row["id"],))
        row["status"] = "running"
        return row


def _heartbeat(job_id: int, stop: threading.Event):
    while not stop.wait(20):
        try:
            db.exec_("update jobs set heartbeat_at=now() where id=%s and status='running'", (job_id,))
        except Exception:
            pass


def run_job(job: dict):
    rep = Reporter(job)
    hb_stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(job["id"], hb_stop), daemon=True).start()
    try:
        out = run_extraction(job, rep) if job["kind"] == "extract" else run_solve_job(job, rep)
        db.exec_("""update jobs set status='done', progress=1, finished_at=now(), eta_s=0,
                       detail = detail || %s::jsonb where id=%s""", (db.dumps({"result": out}), job["id"]))
    except Cancelled:
        db.exec_("update jobs set status='cancelled', finished_at=now(), stage='Cancelled' where id=%s", (job["id"],))
        _restore_run_status(job)
    except Exception as e:
        log.exception("job %s failed", job["id"])
        db.exec_("update jobs set status='failed', finished_at=now(), error=%s where id=%s",
                 (f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}", job["id"]))
        _restore_run_status(job)
    finally:
        hb_stop.set()


def _restore_run_status(job):
    if job["kind"] == "extract":
        db.exec_("""update runs set status = case when exists (select 1 from run_layers where run_id=%s) then 'review'
                    else 'draft' end, updated_at=now() where id=%s""", (job["run_id"], job["run_id"]))
    else:
        db.exec_("""update runs set status = case when latest_solve_id is null then 'review'
                    else (select case when feasible then 'complete' else 'blocked' end from solves where id=latest_solve_id) end,
                    updated_at=now() where id=%s""", (job["run_id"],))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db.init_schema()
    recover_stale(startup=True)
    signal.signal(signal.SIGTERM, lambda *a: _stop.set())
    log.info("worker ready")
    while not _stop.is_set():
        try:
            job = claim()
        except Exception:
            log.exception("claim failed")
            time.sleep(3)
            continue
        if job is None:
            _stop.wait(1.0)
            continue
        log.info("running job %s (%s %s)", job["id"], job["kind"], job["run_id"])
        run_job(job)


if __name__ == "__main__":
    main()
