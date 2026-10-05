import { useEffect, useState } from "react";

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** Re-render every `ms` while `on` — for elapsed-time counters. */
export function useTicker(on: boolean, ms = 1000): number {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!on) return;
    const t = window.setInterval(() => setN((x) => x + 1), ms);
    return () => window.clearInterval(t);
  }, [on, ms]);
  return n;
}
