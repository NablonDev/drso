import type { RunDetail, Solve } from "../api/types";
import { useUi } from "./ui";

/** The route chosen on the map, shared by Route map, Why this route and Export. */
export function useSelectedRoute(run: RunDetail, s: Solve | undefined) {
  const sel = useUi((x) => x.selectedRoute[run.id]);
  const select = useUi((x) => x.selectRoute);
  const ids = s?.result.routes.map((r) => r.id) ?? [];
  const id = sel && ids.includes(sel) ? sel : ids[0] ?? 1;
  return [id, (n: number) => select(run.id, n)] as const;
}
