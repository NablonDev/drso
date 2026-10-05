import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { apiCalls, qk, useApiMutation } from "../api/hooks";
import type { LayerConfig, Mapping, MappingCheck, RunDetail, SourceMode, UploadInfo } from "../api/types";
import { fold, humanClass, int } from "../lib/format";
import { useDebounced } from "../lib/useDebounced";
import { Btn, Dialog, ErrorNote, Note, Spinner } from "./ui";

const NONE = "";
const CONST = "__const__";

function guessTarget(v: string, vocab: string[]): string {
  if (vocab.includes(v)) return v;
  const f = fold(v).replace(/[\s-]+/g, "_");
  return vocab.find((c) => c === f || fold(humanClass(c)) === fold(v)) ?? NONE;
}

function guessField(fields: string[], hints: RegExp): string {
  return fields.find((f) => hints.test(f)) ?? NONE;
}

/** "Map the attributes" — match the user's columns and class values to the Beeline vocabulary. */
export default function MappingDialog({
  open,
  onClose,
  runId,
  layer,
  mode,
  upload,
  initial,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  runId: string;
  layer: LayerConfig;
  mode: SourceMode;
  upload: UploadInfo;
  initial?: Mapping | null;
  onSaved: () => void;
}) {
  const vocab = layer.classes;
  const fields = upload.fields;
  const [subLayer, setSubLayer] = useState(upload.layer);
  const [classField, setClassField] = useState<string>(CONST);
  const [constClass, setConstClass] = useState<string>(vocab[0] ?? "");
  const [nameField, setNameField] = useState<string>(NONE);
  const [confField, setConfField] = useState<string>(NONE);
  const [classMap, setClassMap] = useState<Record<string, string>>({});
  const [rememberAs, setRememberAs] = useState("");
  const [recallKey, setRecallKey] = useState("");
  const [recallMsg, setRecallMsg] = useState<string | null>(null);
  const qc = useQueryClient();

  useEffect(() => {
    if (!open) return;
    setSubLayer(initial?.layer ?? upload.layer);
    const cf = initial?.class_field ?? (vocab.length > 1 ? guessField(fields, /class|type|landuse|lu_|categ|kind/i) : NONE);
    setClassField(cf || CONST);
    setConstClass(initial?.const_class ?? vocab[0] ?? "");
    setNameField(initial?.name_field ?? guessField(fields, /^name|_name|name_en/i));
    setConfField(initial?.conf_field ?? guessField(fields, /conf|score|prob/i));
    setClassMap(initial?.class_map ?? {});
    setRememberAs("");
    setRecallMsg(null);
  }, [open, upload, initial, vocab, fields]);

  const values = useMemo(() => (classField && classField !== CONST ? upload.samples[classField] ?? [] : []), [classField, upload]);
  const effectiveMap = useMemo(() => {
    const m: Record<string, string> = {};
    for (const v of values) {
      const t = classMap[v] ?? guessTarget(v, vocab);
      if (t && t !== v) m[v] = t;
      else if (t === v && !vocab.includes(v)) m[v] = t;
    }
    return m;
  }, [values, classMap, vocab]);

  const mapping: Mapping = useMemo(
    () => ({
      layer: subLayer || null,
      class_field: classField !== CONST ? classField || null : null,
      class_map: classField !== CONST ? effectiveMap : {},
      const_class: classField === CONST ? constClass : null,
      name_field: nameField || null,
      conf_field: confField || null,
    }),
    [subLayer, classField, effectiveMap, constClass, nameField, confField],
  );
  const debounced = useDebounced(mapping, 350);
  const [check, setCheck] = useState<{ res?: MappingCheck; err?: unknown; busy: boolean }>({ busy: false });
  useEffect(() => {
    if (!open) return;
    const ctl = new AbortController();
    setCheck((c) => ({ ...c, busy: true }));
    apiCalls
      .mappingCheck(upload.id, layer.key, debounced, null, ctl.signal)
      .then((res) => setCheck({ res, busy: false }))
      .catch((err: unknown) => {
        if (!ctl.signal.aborted) setCheck({ err, busy: false });
      });
    return () => ctl.abort();
  }, [open, debounced, upload.id, layer.key]);

  const save = useApiMutation<RunDetail, void>({
    mutationFn: async () => {
      if (rememberAs.trim()) await apiCalls.mappingCheck(upload.id, layer.key, mapping, rememberAs.trim());
      return apiCalls.setSource(runId, layer.key, mode, upload.id, mapping);
    },
    silent: true,
    onSuccess: (r) => {
      qc.setQueryData(qk.run(runId), r);
      void qc.invalidateQueries({ queryKey: qk.layers(runId) });
      void qc.invalidateQueries({ queryKey: qk.runs });
      onSaved();
    },
  });

  const recall = async () => {
    setRecallMsg(null);
    try {
      const r = await api<{ mapping: Mapping | null }>(`/api/mappings/${encodeURIComponent(recallKey.trim())}`);
      if (!r.mapping) {
        setRecallMsg("No mapping saved under that name.");
        return;
      }
      const m = r.mapping;
      if (m.layer) setSubLayer(m.layer);
      setClassField(m.class_field || CONST);
      if (m.const_class) setConstClass(m.const_class);
      setNameField(m.name_field ?? NONE);
      setConfField(m.conf_field ?? NONE);
      setClassMap(m.class_map ?? {});
      setRecallMsg("Mapping loaded.");
    } catch (e) {
      setRecallMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const unknown = check.res?.unknown_values ?? [];
  return (
    <Dialog open={open} onClose={onClose} id="mapDlg" labelledBy="mapDlgT">
      <h3 id="mapDlgT">Map the attributes — {layer.name}</h3>
      <p>
        A file from outside this system will not use the column names Beeline expects. Match them once and the mapping can be
        remembered for that source.
      </p>
      <div className="hint" style={{ marginBottom: 10 }}>
        <span className="filechip">{upload.filename}</span> · {int(upload.feature_count)} features · {upload.geometry_type ?? "geometry"} ·{" "}
        {upload.crs ?? "CRS unknown"}
      </div>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>Beeline needs</th>
              <th>Your column</th>
            </tr>
          </thead>
          <tbody>
            {upload.layers.length > 1 ? (
              <tr>
                <td>Layer in the file</td>
                <td>
                  <select value={subLayer} onChange={(e) => setSubLayer(e.target.value)} aria-label="Layer in the file">
                    {upload.layers.map((l) => (
                      <option key={l}>{l}</option>
                    ))}
                  </select>
                  {subLayer !== upload.layer ? (
                    <div className="hint">Columns below are listed from “{upload.layer}”.</div>
                  ) : null}
                </td>
              </tr>
            ) : null}
            <tr>
              <td>
                Feature class <span className="pill p-nogo">required</span>
              </td>
              <td>
                <select value={classField} onChange={(e) => setClassField(e.target.value)} aria-label="Feature class column">
                  <option value={CONST}>— one class for every feature —</option>
                  {fields.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
                {classField === CONST ? (
                  <select value={constClass} onChange={(e) => setConstClass(e.target.value)} style={{ marginTop: 6 }} aria-label="Class for every feature">
                    {vocab.map((c) => (
                      <option key={c} value={c}>
                        {humanClass(c)}
                      </option>
                    ))}
                  </select>
                ) : null}
              </td>
            </tr>
            <tr>
              <td>Name</td>
              <td>
                <select value={nameField} onChange={(e) => setNameField(e.target.value)} aria-label="Name column">
                  <option value={NONE}>— none —</option>
                  {fields.map((f) => (
                    <option key={f}>{f}</option>
                  ))}
                </select>
              </td>
            </tr>
            <tr>
              <td>Confidence</td>
              <td>
                <select value={confField} onChange={(e) => setConfField(e.target.value)} aria-label="Confidence column">
                  <option value={NONE}>— none (1.00) —</option>
                  {fields.map((f) => (
                    <option key={f}>{f}</option>
                  ))}
                </select>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {values.length ? (
        <div className="tw" style={{ marginTop: 10, border: "1px solid var(--rule)", borderRadius: 8 }}>
          <table>
            <thead>
              <tr>
                <th>Value in {classField}</th>
                <th>Beeline class</th>
              </tr>
            </thead>
            <tbody>
              {values.map((v) => {
                const t = classMap[v] ?? guessTarget(v, vocab);
                return (
                  <tr key={v}>
                    <td className="mono" style={{ fontSize: 12 }}>
                      {v}
                    </td>
                    <td>
                      <select value={t} onChange={(e) => setClassMap((m) => ({ ...m, [v]: e.target.value }))} aria-label={`Class for ${v}`}
                        style={!t ? { borderColor: "var(--cross)" } : undefined}>
                        <option value={NONE}>— unrecognised —</option>
                        {vocab.map((c) => (
                          <option key={c} value={c}>
                            {humanClass(c)}
                          </option>
                        ))}
                      </select>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      <div style={{ marginTop: 12 }}>
        {check.busy ? (
          <div className="hint">
            <Spinner /> Checking the mapping…
          </div>
        ) : check.err ? (
          <ErrorNote error={check.err} />
        ) : check.res && unknown.length ? (
          <Note kind="warn">
            <b>
              {unknown.length} of {values.length} class values are unrecognised
            </b>{" "}
            — <span className="mono">{unknown.slice(0, 8).join(", ")}{unknown.length > 8 ? "…" : ""}</span>. Assign each to a Beeline class,
            or they will be treated as <b>{humanClass(check.res.fallback_class)}</b>.
          </Note>
        ) : check.res && !check.res.ok ? (
          <Note kind="stop">Targets not in the vocabulary: {check.res.bad_targets.join(", ")}</Note>
        ) : check.res ? (
          <Note kind="ok">Every value maps to a Beeline class.</Note>
        ) : null}
      </div>

      <div className="grid2" style={{ marginTop: 12, gap: 10 }}>
        <label className="fld">
          <span>Remember this mapping as (optional)</span>
          <input type="text" value={rememberAs} onChange={(e) => setRememberAs(e.target.value)} placeholder="aesl_asset_register" />
        </label>
        <div>
          <label className="fld">
            <span>Or load a saved mapping</span>
            <div className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
              <input type="text" value={recallKey} onChange={(e) => setRecallKey(e.target.value)} placeholder="name" />
              <button className="mini" disabled={!recallKey.trim()} onClick={() => void recall()}>
                Load
              </button>
            </div>
          </label>
          {recallMsg ? <div className="hint">{recallMsg}</div> : null}
        </div>
      </div>

      <ErrorNote error={save.error} />
      <div className="row" style={{ justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
        <button className="btn ghost" onClick={onClose}>
          Cancel
        </button>
        <Btn pending={save.isPending} disabled={Boolean(check.res && !check.res.ok)} onClick={() => save.mutate()}>
          Map and save
        </Btn>
      </div>
    </Dialog>
  );
}
