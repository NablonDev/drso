// Formatting helpers shared by every screen.

export const CR = 1e7;

/** ₹ Cr with 1–2 decimals: two below ₹100 Cr, one above. */
export function crore(rs: number | string | null | undefined, opts: { digits?: number; unit?: boolean } = {}): string {
  if (rs === null || rs === undefined || rs === "") return "—";
  const v = Number(rs) / CR;
  if (!Number.isFinite(v)) return "—";
  const d = opts.digits ?? (Math.abs(v) >= 100 ? 1 : 2);
  const s = v.toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
  return opts.unit === false ? s : `₹${s} Cr`;
}

export function num(v: number | null | undefined, d = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
}

export function int(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return Math.round(v).toLocaleString("en-IN");
}

/** minutes → "4h 29m" / "12m" (the prototype's hm()) */
export function hm(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined || !Number.isFinite(minutes)) return "—";
  const m = Math.max(0, Math.round(minutes));
  const h = Math.floor(m / 60);
  const x = m % 60;
  if (h) return `${h}h ${String(x).padStart(2, "0")}m`;
  return m < 1 ? "<1m" : `${x}m`;
}

export function seconds(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "—";
  if (s < 60) return `${Math.max(1, Math.round(s))}s`;
  return hm(s / 60);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "4 Sep 2026" from ISO date/time; "—" if missing */
export function date(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** "Today, 11:04" / "20 Sep, 11:04" in local time */
export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const now = new Date();
  const hhmm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  if (d.toDateString() === now.toDateString()) return `Today, ${hhmm}`;
  const y = d.getFullYear() === now.getFullYear() ? "" : ` ${d.getFullYear()}`;
  return `${d.getDate()} ${MONTHS[d.getMonth()]}${y}, ${hhmm}`;
}

export function clock(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export function elapsedMin(fromIso: string | null | undefined, toIso?: string | null): number | null {
  if (!fromIso) return null;
  const a = new Date(fromIso).getTime();
  const b = toIso ? new Date(toIso).getTime() : Date.now();
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, (b - a) / 60000);
}

export const CIRCUIT_SHORT: Record<string, string> = { sc: "S/C", dc: "D/C", qc: "Q/C" };
export const CIRCUIT_LONG: Record<string, string> = { sc: "Single circuit", dc: "Double circuit", qc: "Quad circuit" };

/** The corridor's name as the user gave it; older runs fall back to "A → B", then the run id. */
export function corridorName(r: { name?: string | null; id: string; a_name?: string | null; b_name?: string | null }): string {
  if (r.name && r.name.trim()) return r.name.trim();
  if (r.a_name || r.b_name) return `${r.a_name || "A"} → ${r.b_name || "B"}`;
  return r.id;
}

export function lineSpec(kv: number, circuit: string): string {
  return `${kv} kV ${CIRCUIT_SHORT[circuit] ?? circuit}`;
}

export function km(v: number | null | undefined, d = 1): string {
  return v === null || v === undefined ? "—" : `${num(v, d)} km`;
}

/** km chainage like the prototype's "km 14.7" */
export function chainage(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `km ${num(v, 1)}`;
}

export function pct(v: number | null | undefined, d = 0): string {
  return v === null || v === undefined ? "—" : `${num(v, d)}%`;
}

export function conf(v: number | null | undefined): string {
  return v === null || v === undefined ? "n/a" : v.toFixed(2);
}

export function area(m2: number | null | undefined): string {
  if (m2 === null || m2 === undefined || !Number.isFinite(m2)) return "—";
  if (m2 >= 10000) return `${num(m2 / 10000, m2 >= 1e6 ? 0 : 1)} ha`;
  return `${int(m2)} m²`;
}

export function length(m: number | null | undefined): string {
  if (m === null || m === undefined || !Number.isFinite(m)) return "—";
  return m >= 1000 ? `${num(m / 1000, 2)} km` : `${int(m)} m`;
}

export function plural(n: number, one: string, many?: string): string {
  return `${n} ${n === 1 ? one : (many ?? `${one}s`)}`;
}

export function humanClass(c: string | null | undefined): string {
  if (!c) return "—";
  const s = c.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** strip diacritics so "Gujarāt" compares equal to "gujarat" */
export function fold(s: string | null | undefined): string {
  return (s ?? "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();
}

export function fmtLatLon(lat: number, lon: number): string {
  return `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}
