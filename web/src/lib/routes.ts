/** The engine's four routes (config/engine.yaml routes), in solve order (13 Oct 2026). */
export const ROUTE_NAMES: [number, string][] = [
  [1, "Least cost"], [2, "Minimum deviation from the bee-line"], [3, "Relaxed route"], [4, "Second least cost"],
];

/** Toggle one route in the selection: route 4 (second least cost, kept apart from route 1) needs route 1;
 *  the selection never empties. */
export function toggleRouteId(sel: number[], id: number): number[] {
  let next = sel.includes(id) ? sel.filter((x) => x !== id) : [...sel, id];
  if (id === 4 && next.includes(4) && !next.includes(1)) next = [...next, 1];
  if (id === 1 && !next.includes(1)) next = next.filter((x) => x !== 4);
  if (next.length === 0) return sel;
  return [...next].sort((a, b) => a - b);
}
