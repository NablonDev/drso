"""Postgres/PostGIS access: a small connection pool and the schema.

Single-GPU rule (Vivek, 5 Oct 2026): extraction and solve never queue behind each other.
The partial unique index `one_active_job` makes a second active job impossible at the database
level, so two clicks racing each other cannot both start.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from . import settings

log = logging.getLogger(__name__)

SCHEMA = """
create extension if not exists postgis;

create sequence if not exists run_seq start 1;

create table if not exists runs (
  id text primary key,
  family text not null,
  rev int not null default 1,
  parent_id text references runs(id) on delete set null,
  note text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  a_lon double precision not null, a_lat double precision not null, a_name text,
  b_lon double precision not null, b_lat double precision not null, b_name text,
  kv int not null, circuit text not null, band_km double precision not null,
  routes_wanted int not null default 4,
  sources jsonb not null default '{}'::jsonb,
  rule_overrides jsonb not null default '{}'::jsonb,
  cost_overrides jsonb not null default '{}'::jsonb,
  imagery jsonb,
  accept_partial boolean not null default false,
  reuse_from text,
  status text not null default 'draft',
  latest_solve_id bigint,
  geo jsonb
);

create table if not exists jobs (
  id bigserial primary key,
  run_id text not null references runs(id) on delete cascade,
  kind text not null check (kind in ('extract', 'solve')),
  status text not null check (status in ('queued', 'running', 'done', 'failed', 'cancelled')),
  progress double precision not null default 0,
  stage text,
  message text,
  eta_s double precision,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz,
  heartbeat_at timestamptz,
  cancel_requested boolean not null default false,
  error text,
  detail jsonb not null default '{}'::jsonb
);
create unique index if not exists one_active_job on jobs ((true)) where status in ('queued', 'running');
create index if not exists jobs_run on jobs(run_id, created_at desc);

create table if not exists layer_versions (
  id bigserial primary key,
  run_id text references runs(id) on delete cascade,
  layer_key text not null,
  mode text not null,
  producer text,
  source text,
  file_name text,
  model text,
  model_version text,
  image_source text,
  image_date text,
  model_run_at timestamptz,
  confidence double precision,
  feature_count int not null default 0,
  coverage geometry(MultiPolygon, 4326),
  status text not null default 'done',
  error text,
  created_at timestamptz not null default now(),
  stats jsonb not null default '{}'::jsonb
);

create table if not exists run_layers (
  run_id text not null references runs(id) on delete cascade,
  layer_key text not null,
  layer_version_id bigint not null references layer_versions(id) on delete cascade,
  carried boolean not null default false,
  stale boolean not null default false,
  primary key (run_id, layer_key)
);

create table if not exists features (
  id bigserial primary key,
  layer_version_id bigint not null references layer_versions(id) on delete cascade,
  class text,
  name text,
  confidence real,
  src_id text,
  origin text,
  geom geometry(Geometry, 4326) not null
);
alter table features add column if not exists attrs jsonb;
create index if not exists features_geom on features using gist (geom);
create index if not exists features_lv on features (layer_version_id);

create table if not exists solves (
  id bigserial primary key,
  run_id text not null references runs(id) on delete cascade,
  job_id bigint,
  created_at timestamptz not null default now(),
  feasible boolean,
  result jsonb,
  manifest jsonb,
  out_dir text
);

create table if not exists uploads (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  filename text not null,
  path text not null,
  layers jsonb,
  info jsonb
);

create table if not exists attr_mappings (
  source_key text primary key,
  layer_key text not null,
  mapping jsonb not null,
  updated_at timestamptz not null default now()
);

create table if not exists procurement_requests (
  id bigserial primary key,
  run_id text references runs(id) on delete cascade,
  created_at timestamptz not null default now(),
  request jsonb not null,
  status text not null default 'stub'
);
"""

_pool: ConnectionPool | None = None


def _conninfo() -> str:
    url = settings.DB_URL
    return url.replace("postgresql+psycopg://", "postgresql://")


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(_conninfo(), min_size=1, max_size=10, kwargs={"row_factory": dict_row},
                               open=True)
    return _pool


@contextmanager
def conn():
    with pool().connection() as c:
        yield c


def init_schema():
    # API processes and the worker start together on a fresh database; "if not exists" is not
    # race-free in Postgres, so schema creation is serialised with a transaction-level lock.
    with conn() as c:
        with c.transaction():
            c.execute("select pg_advisory_xact_lock(724001)")
            c.execute(SCHEMA)
    from . import osm_power
    osm_power.init_schema()


def J(v):
    return Jsonb(v, dumps=dumps)


def one(sql: str, params=None) -> dict | None:
    with conn() as c:
        return c.execute(sql, params or ()).fetchone()


def all_(sql: str, params=None) -> list[dict]:
    with conn() as c:
        return c.execute(sql, params or ()).fetchall()


def exec_(sql: str, params=None):
    with conn() as c:
        c.execute(sql, params or ())


def dumps(v) -> str:
    return json.dumps(v, default=str)
