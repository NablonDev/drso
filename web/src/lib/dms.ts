// Coordinate parsing: decimal degrees, or DMS like 22°18'25.9"N 73°10'52.3"E.

const DMS_ONE = /(-?\d+(?:\.\d+)?)\s*[°º d]\s*(?:(\d+(?:\.\d+)?)\s*['′’ m]\s*)?(?:(\d+(?:\.\d+)?)\s*(?:["″”]|''|s)?\s*)?([NSEW])?/i;

function dmsToDec(d: string, m?: string, s?: string, hemi?: string): number {
  let v = Math.abs(parseFloat(d)) + (m ? parseFloat(m) / 60 : 0) + (s ? parseFloat(s) / 3600 : 0);
  if (d.trim().startsWith("-")) v = -v;
  if (hemi && /[SW]/i.test(hemi)) v = -Math.abs(v);
  return v;
}

/** Parse a single DMS or decimal value. Returns null when it is not a coordinate. */
export function parseCoord(text: string): number | null {
  const t = text.trim();
  if (!t) return null;
  if (/^-?\d+(\.\d+)?$/.test(t)) return parseFloat(t);
  const m = t.match(DMS_ONE);
  if (!m || !/[°º'′’"″NSEW]/i.test(t)) return null;
  return dmsToDec(m[1], m[2], m[3], m[4]);
}

/**
 * Parse a pasted pair: `22°18'25.9"N 73°10'52.3"E`, `22.3072, 73.1812`, `22.3072 N 73.1812 E`.
 * Returns [lat, lon] or null.
 */
export function parsePair(text: string): [number, number] | null {
  const t = text.trim();
  if (!t) return null;
  // decimal pair
  const dec = t.match(/^(-?\d+(?:\.\d+)?)\s*°?\s*([NS])?\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*°?\s*([EW])?$/i);
  if (dec) {
    let lat = parseFloat(dec[1]);
    let lon = parseFloat(dec[3]);
    if (dec[2] && /S/i.test(dec[2])) lat = -Math.abs(lat);
    if (dec[4] && /W/i.test(dec[4])) lon = -Math.abs(lon);
    return valid(lat, lon) ? [lat, lon] : null;
  }
  // DMS pair: split at the first N/S hemisphere letter
  const split = t.match(/^(.+?[NS])\s*[,;]?\s*(.+?[EW])$/i);
  if (split) {
    const lat = parseCoord(split[1]);
    const lon = parseCoord(split[2]);
    if (lat !== null && lon !== null && valid(lat, lon)) return [lat, lon];
  }
  return null;
}

function valid(lat: number, lon: number) {
  return Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
}

export function fmtDec(v: number, d = 5): string {
  return v.toFixed(d);
}
