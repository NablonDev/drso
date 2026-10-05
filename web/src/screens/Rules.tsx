import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useRules } from "../api/hooks";
import type { Rule, RuleGroup, RuleParam, RulesResponse } from "../api/types";
import { BestGuess, Btn, Dialog, ErrorNote, Head, Kpi, Loading, Note } from "../components/ui";

type Val = number | string | [number, number];
type Draft = Record<string, string | [string, string]>;

export function fmtParam(p: { type: string; unit: string | null }, v: unknown): string {
  const u = p.unit ? ` ${p.unit}` : "";
  if (Array.isArray(v)) return `${v[0]}–${v[1]}${u}`;
  if (p.type === "sel") return String(v);
  return `${v}${u}`;
}

function same(a: unknown, b: unknown) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function toDraft(params: RuleParam[], from: "value" | "default"): Draft {
  const d: Draft = {};
  for (const p of params) {
    const v = p[from];
    d[p.key] = Array.isArray(v) ? [String(v[0]), String(v[1])] : String(v);
  }
  return d;
}

function draftValue(p: RuleParam, d: Draft): Val | string {
  const x = d[p.key];
  if (p.type === "rng" && Array.isArray(x)) {
    const a = Number(x[0]);
    const b = Number(x[1]);
    return x[0].trim() === "" || x[1].trim() === "" || Number.isNaN(a) || Number.isNaN(b) ? (x as unknown as string) : [a, b];
  }
  if (p.type === "num") {
    const s = String(x).trim();
    const n = Number(s);
    return s === "" || Number.isNaN(n) ? s : n;
  }
  return String(x);
}

function draftDirty(p: RuleParam, d: Draft) {
  return !same(draftValue(p, d), p.default);
}

function RuleDialog({
  runId,
  rule,
  statutory,
  onClose,
}: {
  runId: string;
  rule: Rule;
  statutory: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState<Draft>(() => toDraft(rule.params, "value"));
  const [apply, setApply] = useState(rule.enabled);
  const [why, setWhy] = useState(rule.reason ?? "");
  useEffect(() => {
    setDraft(toDraft(rule.params, "value"));
    setApply(rule.enabled);
    setWhy(rule.reason ?? "");
  }, [rule]);

  const nDirty = rule.params.filter((p) => draftDirty(p, draft)).length;
  const anyCh = nDirty > 0 || !apply;
  const save = useApiMutation<RulesResponse, void>({
    mutationFn: () => {
      const params: Record<string, unknown> = {};
      for (const p of rule.params) {
        const v = draftValue(p, draft);
        if (p.type === "num" && typeof v !== "number") throw new Error(`${p.label} must be a number.`);
        if (p.type === "rng" && !(Array.isArray(v) && typeof v[0] === "number" && typeof v[1] === "number"))
          throw new Error(`${p.label} needs two numbers.`);
        params[p.key] = v;
      }
      return apiCalls.setRule(runId, rule.id, { enabled: apply, params, reason: why.trim() || null });
    },
    silent: true,
    onSuccess: (r) => {
      qc.setQueryData(qk.rules(runId), r);
      void qc.invalidateQueries({ queryKey: qk.run(runId) });
      onClose();
    },
  });

  const field = (p: RuleParam) => {
    const ch = draftDirty(p, draft) ? " ch" : "";
    if (p.type === "rng") {
      const v = draft[p.key] as [string, string];
      return (
        <span className="rng">
          <span className={`inp${ch}`} style={{ flex: 1 }}>
            <input type="number" step="any" value={v[0]} aria-label={`${p.label} minimum`}
              onChange={(e) => setDraft((d) => ({ ...d, [p.key]: [e.target.value, (d[p.key] as [string, string])[1]] }))} />
          </span>
          <span>to</span>
          <span className={`inp${ch}`} style={{ flex: 1 }}>
            <input type="number" step="any" value={v[1]} aria-label={`${p.label} maximum`}
              onChange={(e) => setDraft((d) => ({ ...d, [p.key]: [(d[p.key] as [string, string])[0], e.target.value] }))} />
            <i>{p.unit ?? ""}</i>
          </span>
        </span>
      );
    }
    if (p.type === "sel") {
      return (
        <span className={`inp${ch}`}>
          <select value={draft[p.key] as string} aria-label={p.label} onChange={(e) => setDraft((d) => ({ ...d, [p.key]: e.target.value }))}>
            {(p.options ?? []).map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        </span>
      );
    }
    return (
      <span className={`inp${ch}`}>
        <input type="number" step="any" value={draft[p.key] as string} aria-label={p.label}
          onChange={(e) => setDraft((d) => ({ ...d, [p.key]: e.target.value }))} />
        {p.unit ? <i>{p.unit}</i> : null}
      </span>
    );
  };

  return (
    <>
      <h3 id="ruleDlgT">{rule.name}</h3>
      <p className="hint" style={{ marginBottom: 12 }}>
        {statutory ? <span className="pill p-lock" style={{ marginRight: 6 }}>statutory</span> : null}
        {rule.note}
      </p>
      <div className="tw" style={{ border: "1px solid var(--rule)", borderRadius: 8, marginBottom: 12 }}>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Default</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody id="rdRows">
            {rule.params.map((p) => (
              <tr key={p.key}>
                <td style={{ width: "38%" }}>
                  <b style={{ fontSize: 12.5, fontWeight: 500 }}>{p.label}</b>
                  {p.best_guess ? <BestGuess /> : null}
                </td>
                <td className="mono" style={{ width: "22%", color: "var(--faint)", fontSize: 12 }}>{fmtParam(p, p.default)}</td>
                <td>{field(p)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <label className="row" style={{ gap: 9, cursor: "pointer", marginBottom: 10 }}>
        <input type="checkbox" checked={apply} onChange={(e) => setApply(e.target.checked)} style={{ width: "auto", accentColor: "var(--accent)" }} />
        <span style={{ fontSize: 13 }}>Apply this rule to the route</span>
      </label>
      {anyCh ? (
        <label className="fld" style={{ marginBottom: 4 }}>
          <span>Why? (required — tagged as an AESL override in the report)</span>
          <input type="text" id="rdWhy" value={why} onChange={(e) => setWhy(e.target.value)} placeholder="Client agreed on the 14 Sep call" />
        </label>
      ) : null}
      {anyCh && statutory ? (
        <div className="note n-stop" style={{ marginTop: 10, fontSize: 12 }}>
          <div>
            This is a statutory minimum. Saving a different value is permitted, but the export is watermarked and the report names the
            clause it departs from.
          </div>
        </div>
      ) : null}
      {save.error ? (
        <div className="note n-stop" role="alert" style={{ marginTop: 10 }} data-testid="rule-error">
          <div>{save.error.message}</div>
        </div>
      ) : null}
      <div className="row" style={{ justifyContent: "space-between", gap: 8, marginTop: 14 }}>
        <button className="btn ghost" onClick={() => {
          setDraft(toDraft(rule.params, "default"));
          setApply(true);
          setWhy("");
        }}>
          Reset to defaults
        </button>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn ghost" onClick={onClose}>Cancel</button>
          <Btn pending={save.isPending} onClick={() => save.mutate()} id="rdSave">
            {nDirty ? `Save ${nDirty} change${nDirty > 1 ? "s" : ""}` : "Save"}
          </Btn>
        </div>
      </div>
    </>
  );
}

export default function Rules() {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const q = useRules(run.id);
  const [params] = useSearchParams();
  const [search, setSearch] = useState(params.get("q") ?? "");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [edit, setEdit] = useState<{ rule: Rule; statutory: boolean } | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);

  const reset = useApiMutation<RulesResponse, void>({
    mutationFn: () => apiCalls.resetRules(run.id),
    errorTitle: "Rules not reset",
    onSuccess: (r) => {
      qc.setQueryData(qk.rules(run.id), r);
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
      setConfirmReset(false);
    },
  });

  const groups = useMemo(() => q.data?.groups ?? [], [q.data]);
  const ql = search.trim().toLowerCase();
  const filtered = useMemo(
    () =>
      groups.map((g) => ({
        g,
        rules: g.rules.filter(
          (r) => !ql || `${r.name} ${r.note ?? ""} ${r.params.map((p) => p.label).join(" ")}`.toLowerCase().includes(ql),
        ),
      })),
    [groups, ql],
  );

  if (q.isPending) return <Loading text="Loading rules…" />;
  if (q.isError) return <ErrorNote error={q.error} retry={() => void q.refetch()} />;
  const d = q.data;
  const nOvr = groups.reduce((a, g) => a + g.rules.filter((r) => r.n_changed > 0 || !r.enabled).length, 0);
  const isOpen = (g: RuleGroup, i: number) => (ql ? true : (open[g.key] ?? i < 2));

  return (
    <section>
      <Head title="Rules">
        Pre-filled with the default rule set ({d.version}) for {run.kv} kV. Only intervene where the AESL standard differs — every value
        you change is tagged as an AESL override and follows the run into the report.
      </Head>

      <div className="row" style={{ marginBottom: 12, gap: 8 }}>
        <input type="text" id="ruleSearch" placeholder="Find a rule — railway, village, forest, span…" style={{ maxWidth: 300 }}
          value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find a rule" />
        <div className="spacer" />
        <Btn ghost pending={reset.isPending} disabled={Boolean(run.job) || (d.n_changed === 0 && d.n_off === 0)}
          onClick={() => setConfirmReset(true)}>
          Reset everything to defaults
        </Btn>
      </div>

      {run.job ? (
        <Note kind="warn" style={{ marginBottom: 14 }}>A job is running for this run — rule changes are refused until it finishes.</Note>
      ) : null}

      <div className="kpis" style={{ marginBottom: 14 }}>
        <Kpi k="Rules applied" v={d.n_rules - d.n_off} color="var(--go)" />
        <Kpi k="Parameters shown" v={d.n_params} />
        <Kpi k="Parameters changed" v={d.n_changed} color="var(--changed)" />
        <Kpi k="AESL overrides" v={nOvr} color="var(--changed)" />
        <Kpi k="Switched off" v={d.n_off} color="var(--cross)" />
      </div>

      <Note kind="info" style={{ marginBottom: 14 }}>
        <b>A rule is a set of parameters, not one number.</b> The railway rule alone carries six — tower setback, the structure-height
        allowance that produces it, the crossing angle range, the span cap, the percentage-of-design-span alternative, and the tower
        type. The list shows all of them at their current values; <b>Change</b> opens them all for editing side by side with their
        defaults.
      </Note>
      <Note kind="ok" style={{ marginBottom: 14 }}>
        <b>Land classes are go or no-go, nothing in between.</b> There are no avoidance sliders and no weights on this screen. How
        strongly the engine prefers one class over another is an internal model parameter, not a business decision — the drafter
        sets what the route may and may not enter, and the engine handles the rest.
      </Note>

      <div id="ruleGroups">
        {filtered.every((f) => f.rules.length === 0) ? (
          <div className="card">
            <div className="cb">
              <p style={{ color: "var(--muted)", fontSize: 13 }}>No rule matches that search.</p>
            </div>
          </div>
        ) : null}
        {filtered.map(({ g, rules }, gi) =>
          rules.length === 0 ? null : (
            <div key={g.key} className={`acc ${isOpen(g, gi) ? "open" : ""}`}>
              <button className="acch" aria-expanded={isOpen(g, gi)} onClick={() => setOpen((o) => ({ ...o, [g.key]: !isOpen(g, gi) }))}>
                <span className={`pill ${g.statutory ? "p-lock" : g.key === "nogo" ? "p-nogo" : g.key === "cross" ? "p-cross" : "p-go"}`}>{g.title}</span>
                <span className="cnt">{g.rules.length} rules</span>
                <span style={{ fontSize: 12.5, color: "var(--muted)" }}>{g.desc}</span>
                <span className="car">›</span>
              </button>
              <div className="accb">
                {rules.map((r) => (
                  <div key={r.id} className={`rule2 ${r.enabled ? "" : "disabled"} ${r.n_changed ? "dirty" : ""}`} data-rule={r.id}>
                    <div className="rhead">
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="t">
                          {r.name}{" "}
                          {g.statutory ? <span className="pill p-lock">statutory</span> : null}{" "}
                          {r.n_changed ? <span className="pill p-chg">AESL override · {r.n_changed} field{r.n_changed > 1 ? "s" : ""}</span> : null}{" "}
                          {!r.enabled ? <span className="pill p-nogo">not applied</span> : null}
                        </div>
                        <div className="s">{r.note}</div>
                        {r.reason && (r.n_changed || !r.enabled) ? (
                          <div className="s" style={{ color: "var(--changed)" }}>
                            “{r.reason}”{r.rev ? ` · since rev ${r.rev}` : ""}
                          </div>
                        ) : null}
                        <div className="pvs">
                          {r.params.map((p) => (
                            <span key={p.key} className={`pv ${p.changed ? "pvch" : ""}`}>
                              <i>{p.label}</i>
                              {fmtParam(p, p.value)}
                              {p.best_guess ? <BestGuess /> : null}
                            </span>
                          ))}
                        </div>
                      </div>
                      <button className="mini" onClick={() => setEdit({ rule: r, statutory: g.statutory })} aria-label={`Change ${r.name}`}>
                        Change
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ),
        )}
      </div>

      <div className="actions">
        <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/sources`)}>Back</button>
        <button className="btn" onClick={() => nav(`/runs/${run.id}/costs`)}>Next: costs</button>
      </div>

      <Dialog open={Boolean(edit)} onClose={() => setEdit(null)} id="ruleDlg" labelledBy="ruleDlgT">
        {edit ? <RuleDialog runId={run.id} rule={edit.rule} statutory={edit.statutory} onClose={() => setEdit(null)} /> : null}
      </Dialog>

      <Dialog open={confirmReset} onClose={() => setConfirmReset(false)} labelledBy="rstT">
        <h3 id="rstT">Reset every rule to its default?</h3>
        <p>
          {d.n_changed} changed parameter{d.n_changed === 1 ? "" : "s"} and {d.n_off} switched-off rule{d.n_off === 1 ? "" : "s"} go back
          to the default set, and their reasons are dropped from this run.
        </p>
        <ErrorNote error={reset.error} />
        <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
          <button className="btn ghost" onClick={() => setConfirmReset(false)}>Cancel</button>
          <Btn pending={reset.isPending} onClick={() => reset.mutate()}>Reset everything</Btn>
        </div>
      </Dialog>
    </section>
  );
}
