import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useCosts } from "../api/hooks";
import type { CostsResponse } from "../api/types";
import { BestGuess, ErrorNote, Head, Loading, Note } from "../components/ui";
import { num } from "../lib/format";

type Body = Record<string, Record<string, number>>;

/** A number field that saves itself on blur / Enter and shows the save state. */
function NumCell({
  id,
  value,
  dflt,
  label,
  width = 78,
  disabled,
  saving,
  onCommit,
  onDraft,
}: {
  id: string;
  value: number;
  dflt?: number;
  label: string;
  width?: number;
  disabled?: boolean;
  saving?: boolean;
  onCommit: (v: number) => void;
  onDraft?: (v: number | null) => void;
}) {
  const [s, setS] = useState(String(value));
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setS(String(value));
    setErr(null);
  }, [value]);
  const commit = () => {
    const t = s.trim();
    const n = Number(t.replace(/,/g, ""));
    if (t === "" || Number.isNaN(n)) {
      setErr("number");
      return;
    }
    if (n < 0) {
      setErr("≥ 0");
      return;
    }
    setErr(null);
    if (n !== value) onCommit(n);
  };
  const changed = dflt !== undefined && Number(s) !== dflt;
  return (
    <span className={`inp${changed ? " ch" : ""}`} style={{ width, display: "inline-flex", borderColor: err ? "var(--nogo)" : undefined }}>
      <input
        id={id}
        type="text"
        inputMode="decimal"
        className="mono"
        style={{ textAlign: "right" }}
        value={s}
        aria-label={label}
        aria-invalid={Boolean(err)}
        disabled={disabled}
        title={err ? `Must be a number ${err === "≥ 0" ? "of zero or more" : ""}` : dflt !== undefined ? `Default ${dflt}` : undefined}
        onChange={(e) => {
          setS(e.target.value);
          const n = Number(e.target.value);
          onDraft?.(e.target.value.trim() === "" || Number.isNaN(n) ? null : n);
        }}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          if (e.key === "Escape") {
            setS(String(value));
            onDraft?.(null);
          }
        }}
      />
      {saving ? (
        <i>
          <span className="spin" style={{ width: 10, height: 10 }} aria-label="Saving" />
        </i>
      ) : null}
    </span>
  );
}

export default function Costs() {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const q = useCosts(run.id);
  const [rateDraft, setRateDraft] = useState<number | null>(null);
  const [tonDraft, setTonDraft] = useState<Record<string, number | null>>({});
  const [savingKey, setSavingKey] = useState<string | null>(null);

  const put = useApiMutation<CostsResponse, { body: Body; key: string }>({
    mutationFn: ({ body }) => apiCalls.setCosts(run.id, body),
    errorTitle: "Rate not saved",
    onMutate: ({ key }) => setSavingKey(key),
    onSettled: () => setSavingKey(null),
    onSuccess: (r) => {
      qc.setQueryData(qk.costs(run.id), r);
      setRateDraft(null);
      setTonDraft({});
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
    },
    onError: () => {
      setRateDraft(null);
      setTonDraft({});
    },
  });

  if (q.isPending) return <Loading text="Loading cost rates…" />;
  if (q.isError) return <ErrorNote error={q.error} retry={() => void q.refetch()} />;
  const d = q.data;
  const locked = Boolean(run.job);
  const rate = rateDraft ?? d.rate_per_tonne_rs;
  const nCats = d.categories.length;
  const tcc = d.categories.filter((c) => c.status === "tcc");
  const sourced = d.categories.filter((c) => c.status === "sourced");
  const save = (key: string, body: Body) => put.mutate({ key, body });

  return (
    <section>
      <Head title="Costs">
        {nCats === 8 ? "Eight" : nCats} categories of unit rate. These are what the route is optimised against, so a wrong rate moves the
        alignment, not just the total.
      </Head>

      {d.n_tcc > 0 ? (
        <Note kind="warn" style={{ marginBottom: 14 }}>
          <b>Draft rates — not yet signed off.</b> The values below are best-effort figures ({d.version}) pending review by the AESL TCC
          team. Tower, conductor and foundation rates in particular vary by voltage and must come from TCC. Categories still awaiting it
          are marked <span className="pill p-cross">TCC</span>.
        </Note>
      ) : null}
      {locked ? <Note kind="warn" style={{ marginBottom: 14 }}>A job is running for this run — rates are read-only until it finishes.</Note> : null}

      <div className="grid2">
        <div className="card">
          <div className="ch">
            <h3>Tower hardware — rate by type</h3>
            {d.tonnage_best_guess ? <BestGuess /> : null}
            <span className="pill p-cross" style={{ marginLeft: "auto" }}>TCC</span>
          </div>
          <div className="tw">
            <table data-testid="tonnage-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Deviation</th>
                  <th>Weight</th>
                  <th style={{ textAlign: "right" }}>₹ Cr each</th>
                </tr>
              </thead>
              <tbody>
                {d.tonnage.map((t) => {
                  const tonnes = tonDraft[t.type] ?? t.tonnes;
                  return (
                    <tr key={t.type}>
                      <td><b>{t.type}</b></td>
                      <td className="mono">{t.deviation}</td>
                      <td>
                        <span className="row" style={{ gap: 4, flexWrap: "nowrap" }}>
                          <NumCell id={`ton-${t.type}`} label={`Tonnage of type ${t.type}`} value={t.tonnes} dflt={t.default_tonnes} width={84}
                            disabled={locked} saving={savingKey === `ton-${t.type}`}
                            onDraft={(v) => setTonDraft((s) => ({ ...s, [t.type]: v }))}
                            onCommit={(v) => save(`ton-${t.type}`, { tower_tonnage: { [t.type]: v } })} />
                          <span className="hint" style={{ margin: 0 }}>t</span>
                        </span>
                      </td>
                      <td className="mono" style={{ textAlign: "right" }} data-testid={`cr-${t.type}`}>
                        {num((tonnes * rate) / 1e7, 4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="cb" style={{ borderTop: "1px solid var(--rule-2)" }}>
            <div className="row" style={{ gap: 8, marginBottom: 10 }}>
              <label htmlFor="rate" style={{ fontSize: 12.5, color: "var(--ink-2)" }}>Rate per tonne</label>
              <NumCell id="rate" label="Rate per tonne in rupees" value={d.rate_per_tonne_rs}
                dflt={d.categories.find((c) => c.key === "tower")?.items.find((i) => i.key === "rate_per_tonne_rs")?.default}
                width={120} disabled={locked} saving={savingKey === "rate"} onDraft={setRateDraft}
                onCommit={(v) => save("rate", { tower: { rate_per_tonne_rs: v } })} />
              <span className="hint" style={{ margin: 0 }}>₹/t</span>
            </div>
            <Note kind="info">
              Derived as <span className="mono">weight × ₹{num(rate, 0)}/tonne</span> at {run.kv} kV. Edit the rate per tonne and all
              seven move together. Tonnages and the rate per tonne both need TCC confirmation — they change with voltage.
            </Note>
          </div>
        </div>
        <div className="stack">
          <div className="card">
            <div className="ch"><h3>Where these rates come from</h3></div>
            <div className="cb">
              <div className="tw">
                <table>
                  <tbody>
                    {sourced.map((c) => (
                      <tr key={c.key}>
                        <td><span className="pill p-go">sourced</span></td>
                        <td><b>{c.title}</b><div className="hint">{c.desc}</div></td>
                      </tr>
                    ))}
                    {tcc.length ? (
                      <tr>
                        <td><span className="pill p-cross">pending</span></td>
                        <td>
                          <b>{tcc.map((c) => c.title).join(", ")}</b>
                          <div className="hint">Must come from the AESL TCC team — they vary by voltage. Best-effort figures are loaded.</div>
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="ch">
          <h3>The {nCats === 8 ? "eight" : nCats} cost categories</h3>
          <span className="hint" style={{ marginLeft: "auto" }}>every rate editable · {d.n_tcc} of {nCats} awaiting TCC</span>
        </div>
        <div id="costCats">
          {d.categories.map((c, i) => {
            const nChanged = c.items.filter((it) => it.changed).length;
            return (
              <div key={c.key} className="cat" data-cat={c.key}>
                <div className="ct">
                  <span className="pill p-acc">{i + 1}</span>
                  {c.title}
                  {c.status === "tcc" ? <span className="pill p-cross">TCC sign-off needed</span> : <span className="pill p-go">sourced</span>}
                  {nChanged ? <span className="pill p-chg">{nChanged} changed</span> : null}
                  {nChanged ? (
                    <button className="mini" style={{ marginLeft: "auto" }} disabled={locked || put.isPending}
                      onClick={() => save(`reset-${c.key}`, { [c.key]: Object.fromEntries(c.items.map((it) => [it.key, it.default])) })}>
                      {savingKey === `reset-${c.key}` ? <span className="spin" style={{ width: 10, height: 10 }} /> : null} Reset
                    </button>
                  ) : null}
                </div>
                <div className="cs">{c.desc}</div>
                <div className="catrows">
                  {c.items.map((it) => (
                    <div key={it.key} className="catrow">
                      <label className="l" htmlFor={`c-${c.key}-${it.key}`}>
                        {it.label}
                        {it.best_guess ? <BestGuess /> : null}
                      </label>
                      <NumCell id={`c-${c.key}-${it.key}`} label={`${c.title}: ${it.label}`} value={it.value} dflt={it.default} width={86}
                        disabled={locked} saving={savingKey === `${c.key}.${it.key}`}
                        onDraft={c.key === "tower" && it.key === "rate_per_tonne_rs" ? setRateDraft : undefined}
                        onCommit={(v) => save(`${c.key}.${it.key}`, { [c.key]: { [it.key]: v } })} />
                      <span className="u">{it.unit}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <Note kind="info" style={{ marginTop: 14 }}>
        Rates are stored per voltage, so the 400 kV set is separate from this one. A rate left at its draft value still runs — the
        report carries an unsigned-rate notice and names every category it applies to.
      </Note>
      <div className="actions">
        <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/rules`)}>Back</button>
        <button className="btn" onClick={() => nav(`/runs/${run.id}/review`)}>Next: review</button>
      </div>
    </section>
  );
}
