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
  const [savingKey, setSavingKey] = useState<string | null>(null);

  const put = useApiMutation<CostsResponse, { body: Body; key: string }>({
    mutationFn: ({ body }) => apiCalls.setCosts(run.id, body),
    errorTitle: "Rate not saved",
    onMutate: ({ key }) => setSavingKey(key),
    onSettled: () => setSavingKey(null),
    onSuccess: (r) => {
      qc.setQueryData(qk.costs(run.id), r);
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
    },
  });

  if (q.isPending) return <Loading text="Loading cost rates…" />;
  if (q.isError) return <ErrorNote error={q.error} retry={() => void q.refetch()} />;
  const d = q.data;
  const locked = Boolean(run.job);
  const nCats = d.categories.filter((c) => c.items.length > 0).length;
  const tcc = d.categories.filter((c) => c.status === "tcc");
  const sourced = d.categories.filter((c) => c.status === "sourced");
  const save = (key: string, body: Body) => put.mutate({ key, body });

  return (
    <section>
      <Head title="Costs">
        {nCats} categories of unit rate. These are what the route is optimised against, so a wrong rate moves the
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
            <h3>Towers — price by type ({run.kv} kV D/C)</h3>
            <span className="pill p-cross" style={{ marginLeft: "auto" }}>TCC</span>
          </div>
          <div className="tw">
            <table data-testid="tower-price-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Deviation</th>
                  <th>₹ Cr each</th>
                  <th style={{ textAlign: "right" }}>Base foundation ₹ L</th>
                  <th style={{ textAlign: "right" }}>Base area m²</th>
                </tr>
              </thead>
              <tbody>
                {d.towers.map((t) => (
                  <tr key={t.type}>
                    <td><b>{t.type}</b></td>
                    <td className="mono">{t.deviation}</td>
                    <td>
                      <NumCell id={`price-${t.type}`} label={`Price of type ${t.type} in rupees crore`} value={t.price_cr}
                        dflt={t.default_price_cr} width={84} disabled={locked} saving={savingKey === `price-${t.type}`}
                        onCommit={(v) => save(`price-${t.type}`, { tower_price: { [t.type]: v } })} />
                    </td>
                    <td className="mono" style={{ textAlign: "right" }} data-testid={`fdn-${t.type}`}>{num(t.fdn_l, 2)}</td>
                    <td className="mono" style={{ textAlign: "right" }}>{num(t.area_m2, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="cb" style={{ borderTop: "1px solid var(--rule-2)" }}>
            <Note kind="info">
              The price is all-in: steel, erection and a normal foundation. Where the soil needs a dearer foundation, the extra is
              the base foundation × (soil multiplier − 1). Tower-base compensation is the base area × land value × tower-base
              share. Line items add {num(d.line_cr_km, 3)} ₹ Cr per km at this voltage.
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
          <h3>The {nCats} cost categories</h3>
          <span className="hint" style={{ marginLeft: "auto" }}>every rate editable · {d.n_tcc} of {nCats} awaiting TCC</span>
        </div>
        <div id="costCats">
          {d.categories.filter((c) => c.items.length > 0).map((c, i) => {
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
        <button className="btn" onClick={() => nav(`/runs/${run.id}/extracting`)}>Next: extract layers</button>
      </div>
    </section>
  );
}
