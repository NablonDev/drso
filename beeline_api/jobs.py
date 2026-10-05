"""Job control: start (refused while anything runs), progress, heartbeat, cancel."""
from __future__ import annotations

from psycopg.errors import UniqueViolation

from . import db


class Busy(Exception):
    def __init__(self, job: dict | None):
        self.job = job or {}
        if not job:
            super().__init__("Another job is starting. Nothing is queued — try again in a moment.")
            return
        kind = "LULC extraction" if job["kind"] == "extract" else "Route solve"
        eta = f", about {fmt_eta(job.get('eta_s'))} left" if job.get("eta_s") else ""
        stage = f" ({job['stage']})" if job.get("stage") else ""
        other = "a route solve" if job["kind"] == "extract" else "a LULC extraction"
        super().__init__(f"{kind} is running for {job['run_id']}{stage}{eta}. "
                         f"Nothing is queued — start {other} once it finishes.")


class Cancelled(Exception):
    pass


def fmt_eta(s) -> str:
    if s is None:
        return "—"
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{max(1, m)}m"


def active() -> dict | None:
    return db.one("select * from jobs where status in ('queued','running') order by id desc limit 1")


def start(run_id: str, kind: str, detail: dict | None = None) -> dict:
    a = active()
    if a:
        raise Busy(a)
    try:
        return db.one("""insert into jobs (run_id, kind, status, stage, detail)
                         values (%s, %s, 'queued', 'Waiting for the worker', %s) returning *""",
                      (run_id, kind, db.J(detail or {})))
    except UniqueViolation:
        raise Busy(active())


def get(job_id: int) -> dict | None:
    return db.one("select * from jobs where id=%s", (job_id,))


def latest_for(run_id: str, kind: str | None = None) -> dict | None:
    if kind:
        return db.one("select * from jobs where run_id=%s and kind=%s order by id desc limit 1", (run_id, kind))
    return db.one("select * from jobs where run_id=%s order by id desc limit 1", (run_id,))


def request_cancel(job_id: int) -> dict | None:
    return db.one("""update jobs set cancel_requested=true,
                       status = case when status='queued' then 'cancelled' else status end,
                       finished_at = case when status='queued' then now() else finished_at end
                     where id=%s and status in ('queued','running') returning *""", (job_id,))


class Reporter:
    """Progress writer used by the worker; also the cancellation check."""

    def __init__(self, job: dict):
        self.job = job
        self.id = job["id"]
        import time
        self._t0 = time.time()
        self._last = 0.0
        self._time = time

    def __call__(self, frac: float, stage: str | None = None, message: str | None = None, detail: dict | None = None,
                 force: bool = False):
        now = self._time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        frac = max(0.0, min(1.0, float(frac)))
        el = now - self._t0
        eta = el / frac * (1 - frac) if frac > 0.02 else None
        with db.conn() as c:
            c.execute("""update jobs set progress=%s, stage=coalesce(%s, stage), message=coalesce(%s, message),
                           eta_s=%s, heartbeat_at=now(),
                           detail = case when %s::jsonb is null then detail
                                         else (detail || (%s::jsonb - 'layers')) || jsonb_build_object('layers',
                                              coalesce(case when jsonb_typeof(detail->'layers') = 'object' then detail->'layers' end,
                                                       '{}'::jsonb) || coalesce(%s::jsonb->'layers', '{}'::jsonb)) end
                         where id=%s""",
                      (frac, stage, message, eta, db.dumps(detail) if detail else None,
                       db.dumps(detail) if detail else None, db.dumps(detail) if detail else None, self.id))

    def cancelled(self) -> bool:
        r = db.one("select cancel_requested from jobs where id=%s", (self.id,))
        return bool(r and r["cancel_requested"])

    def check(self):
        if self.cancelled():
            raise Cancelled()
