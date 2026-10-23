import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { parsePair } from "../../lib/dms";
import { ESRI_MAX_ZOOM } from "./style";

/**
 * Go to a coordinate (22 Oct 2026): type or paste `22.44097, 72.43361` or `22°26'27.5"N 72°26'01.0"E`, Enter flies the
 * map there and drops a pin with the point's coordinates; × removes the pin. One pin per map.
 */
export function CoordSearch({ map, zoom = 16 }: { map: maplibregl.Map | null; zoom?: number }) {
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [pinned, setPinned] = useState<[number, number] | null>(null);
  const marker = useRef<maplibregl.Marker | null>(null);

  useEffect(() => () => {
    marker.current?.remove();
    marker.current = null;
  }, [map]);

  const go = () => {
    if (!map) return;
    const p = parsePair(text);
    if (!p) {
      setErr("Not a coordinate — try 22.44097, 72.43361 or 22°26'27.5\"N 72°26'01.0\"E");
      return;
    }
    const [lat, lon] = p;
    setErr(null);
    marker.current?.remove();
    const label = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    const popup = new maplibregl.Popup({ offset: 28, closeButton: false }).setText(label);
    marker.current = new maplibregl.Marker({ color: "#D9534F" }).setLngLat([lon, lat]).setPopup(popup).addTo(map);
    marker.current.togglePopup();
    setPinned([lat, lon]);
    map.flyTo({ center: [lon, lat], zoom: Math.min(zoom, ESRI_MAX_ZOOM), essential: true });
  };

  const clear = () => {
    marker.current?.remove();
    marker.current = null;
    setPinned(null);
    setText("");
    setErr(null);
  };

  return (
    <span className="coordsearch" data-testid="coord-search" title={err ?? "Go to a coordinate: decimal degrees or degrees-minutes-seconds"}>
      <input
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          if (err) setErr(null);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") go();
          if (e.key === "Escape") clear();
        }}
        placeholder="Go to lat, lon"
        aria-label="Go to a coordinate"
        aria-invalid={err ? true : undefined}
        disabled={!map}
        data-testid="coord-input"
      />
      <button className="zb" onClick={go} disabled={!map || !text.trim()} aria-label="Go to the coordinate" data-testid="coord-go">Go</button>
      {pinned ? (
        <button className="zb" onClick={clear} aria-label="Remove the pin" title="Remove the pin" data-testid="coord-clear">×</button>
      ) : null}
      {err ? <span className="coorderr" role="alert">{err}</span> : null}
    </span>
  );
}
