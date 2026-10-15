import {
  QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationOptions,
} from "@tanstack/react-query";
import { ApiError, api, errorText } from "./client";
import type { DemCoverage, RouteProfile, ImageryCatalog, ModelStatus, NearestPoint, NearestResult,
  AppConfig,
  BandInfo,
  CompareResponse,
  CostsResponse,
  Feature,
  GeoValidate,
  Health,
  Imagery,
  Job,
  LayersResponse,
  Mapping,
  MappingCheck,
  ProcurementResponse,
  Pt,
  RulesResponse,
  RunDetail,
  RunsResponse,
  Solve,
  SourceMode,
} from "./types";
import { useUi } from "../store/ui";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      refetchOnWindowFocus: false,
      retry: (count, err) => {
        if (err instanceof ApiError && err.status < 500) return false;
        return count < 1;
      },
    },
    mutations: { retry: false },
  },
});

export const qk = {
  config: ["config"] as const,
  health: ["health"] as const,
  runs: ["runs"] as const,
  run: (id: string) => ["run", id] as const,
  layers: (id: string) => ["layers", id] as const,
  rules: (id: string) => ["rules", id] as const,
  costs: (id: string) => ["costs", id] as const,
  solve: (id: string, sid: number | null | undefined) => ["solve", id, sid ?? 0] as const,
  compare: (id: string, sid: number | null | undefined) => ["compare", id, sid ?? 0] as const,
  active: ["jobs", "active"] as const,
  feature: (id: string, fid: number) => ["feature", id, fid] as const,
};

// ---------- queries ----------------------------------------------------------------------

export function useConfig() {
  return useQuery({ queryKey: qk.config, queryFn: () => api<AppConfig>("/api/config"), staleTime: Infinity });
}

export function useHealth() {
  return useQuery({ queryKey: qk.health, queryFn: () => api<Health>("/api/health"), staleTime: 300_000 });
}

export function useActiveJob() {
  return useQuery({
    queryKey: qk.active,
    queryFn: () => api<{ job: Job | null }>("/api/jobs/active"),
    refetchInterval: 4000,
    refetchIntervalInBackground: false,
    staleTime: 2000,
  });
}

export function useRuns() {
  const active = useActiveJob().data?.job;
  return useQuery({
    queryKey: qk.runs,
    queryFn: () => api<RunsResponse>("/api/runs"),
    staleTime: 3000,
    refetchInterval: active ? 4000 : 20_000,
  });
}

export function useRun(id: string | undefined) {
  return useQuery({
    queryKey: qk.run(id ?? ""),
    queryFn: () => api<RunDetail>(`/api/runs/${encodeURIComponent(id!)}`),
    enabled: Boolean(id),
    staleTime: 2000,
    refetchInterval: (q) => {
      const d = q.state.data;
      if (!d) return false;
      if (d.job || d.status === "extracting" || d.status === "solving") return 2000;
      return false;
    },
  });
}

export function useLayers(id: string | undefined, opts: { poll?: boolean; enabled?: boolean } = {}) {
  return useQuery({
    queryKey: qk.layers(id ?? ""),
    queryFn: () => api<LayersResponse>(`/api/runs/${encodeURIComponent(id!)}/layers`),
    enabled: Boolean(id) && opts.enabled !== false,
    staleTime: 10_000,
    refetchInterval: opts.poll ? 5000 : false,
  });
}

export function useRules(id: string | undefined) {
  return useQuery({
    queryKey: qk.rules(id ?? ""),
    queryFn: () => api<RulesResponse>(`/api/runs/${encodeURIComponent(id!)}/rules`),
    enabled: Boolean(id),
    staleTime: 30_000,
  });
}

export function useCosts(id: string | undefined) {
  return useQuery({
    queryKey: qk.costs(id ?? ""),
    queryFn: () => api<CostsResponse>(`/api/runs/${encodeURIComponent(id!)}/costs`),
    enabled: Boolean(id),
    staleTime: 30_000,
  });
}

/** The latest solve. Keyed by solve id: a solve never changes once written. */
export function useDemCoverage(runId: string | undefined) {
  return useQuery({
    queryKey: ["dem-coverage", runId],
    queryFn: () => api<DemCoverage>(`/api/runs/${encodeURIComponent(runId!)}/dem`),
    enabled: Boolean(runId),
    staleTime: Infinity,
  });
}

/** Closest feature of a run layer to each point (one request per route; cached for the session). */
export function useNearest(runId: string | undefined, key: string, points: NearestPoint[] | null, enabled = true) {
  return useQuery({
    queryKey: ["nearest", runId ?? "", key] as const,
    queryFn: () => api<{ results: (NearestResult | null)[] }>(`/api/runs/${encodeURIComponent(runId!)}/nearest`, { method: "POST", json: { points } }),
    enabled: enabled && Boolean(runId) && Boolean(points && points.length),
    staleTime: Infinity,
  });
}

export function useRouteProfile(sid: number | null | undefined, routeId: number | null | undefined, enabled = true) {
  return useQuery({
    queryKey: ["profile", sid ?? 0, routeId ?? 0],
    queryFn: () => api<RouteProfile>(`/api/solves/${sid}/routes/${routeId}/profile`),
    enabled: Boolean(sid && routeId && enabled),
    staleTime: Infinity,
    gcTime: 10 * 60_000,
  });
}

export function useSolve(id: string | undefined, sid: number | null | undefined) {
  return useQuery({
    queryKey: qk.solve(id ?? "", sid),
    queryFn: () => api<Solve>(`/api/runs/${encodeURIComponent(id!)}/solve`),
    enabled: Boolean(id && sid),
    staleTime: Infinity,
    gcTime: 10 * 60_000,
  });
}

export function useCompare(id: string | undefined, sid: number | null | undefined, enabled = true) {
  return useQuery({
    queryKey: qk.compare(id ?? "", sid),
    queryFn: () => api<CompareResponse>(`/api/runs/${encodeURIComponent(id!)}/compare`),
    enabled: Boolean(id && sid) && enabled,
    staleTime: 60_000,
  });
}

export function useFeature(runId: string | undefined, fid: number | null) {
  return useQuery({
    queryKey: qk.feature(runId ?? "", fid ?? -1),
    queryFn: () => api<Feature>(`/api/runs/${encodeURIComponent(runId!)}/features/${fid}`),
    enabled: Boolean(runId) && fid !== null,
    staleTime: Infinity,
  });
}

export function useGeoValidate(a: Pt | null, b: Pt | null, band: number) {
  return useQuery({
    queryKey: ["geo", "validate", a?.lat, a?.lon, b?.lat, b?.lon],
    queryFn: ({ signal }) =>
      api<GeoValidate>("/api/geo/validate", { method: "POST", json: { a, b, band_km: band }, signal }),
    enabled: Boolean(a && b),
    staleTime: 300_000,
    placeholderData: (prev) => prev,
  });
}

export function useGeoBand(a: Pt | null, b: Pt | null, band: number, exclude?: string) {
  return useQuery({
    queryKey: ["geo", "band", a?.lat, a?.lon, b?.lat, b?.lon, band, exclude ?? ""],
    queryFn: ({ signal }) =>
      api<BandInfo>(`/api/geo/band${exclude ? `?exclude=${encodeURIComponent(exclude)}` : ""}`, {
        method: "POST",
        json: { a, b, band_km: band },
        signal,
      }),
    enabled: Boolean(a && b),
    staleTime: 60_000,
    placeholderData: (prev) => prev,
  });
}

// ---------- mutations --------------------------------------------------------------------

/**
 * useMutation with Beeline defaults: never retried, every error is shown as a toast with the
 * server's `detail`, and a 409 "busy" refreshes the global job indicator.
 */
export function useApiMutation<TData, TVars = void>(
  opts: UseMutationOptions<TData, Error, TVars> & { silent?: boolean; errorTitle?: string },
) {
  const qc = useQueryClient();
  const push = useUi((s) => s.push);
  const { silent, errorTitle, onError, ...rest } = opts;
  return useMutation<TData, Error, TVars>({
    ...rest,
    onError: (err, vars, ctx, mctx) => {
      if (err instanceof ApiError && err.busy) {
        void qc.invalidateQueries({ queryKey: qk.active });
        push({ kind: "stop", title: "The GPU is busy", text: err.detail, job: err.job, sticky: true });
      } else if (!silent) {
        push({ kind: "stop", title: errorTitle, text: errorText(err) });
      }
      onError?.(err, vars, ctx, mctx);
    },
  });
}

export function useInvalidateRun() {
  const qc = useQueryClient();
  return (id: string) => {
    void qc.invalidateQueries({ queryKey: qk.run(id) });
    void qc.invalidateQueries({ queryKey: qk.runs });
  };
}

export interface RunCreateBody {
  name: string;
  a: Pt;
  b: Pt;
  kv: number;
  /** always "dc" from the UI (double circuit); the server defaults to it too */
  circuit?: string;
  band_km: number;
  routes_wanted?: number;
  routes_selected?: number[];
  note?: string | null;
  reuse_from?: string | null;
}

export const apiCalls = {
  createRun: (body: RunCreateBody) => api<RunDetail>("/api/runs", { method: "POST", json: body }),
  patchRun: (id: string, body: Partial<RunCreateBody> & { accept_partial?: boolean }) =>
    api<RunDetail>(`/api/runs/${encodeURIComponent(id)}`, { method: "PATCH", json: body }),
  deleteRun: (id: string) => api<{ deleted: string }>(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  copyRun: (id: string, note: string | null) =>
    api<RunDetail>(`/api/runs/${encodeURIComponent(id)}/copy`, { method: "POST", json: { note } }),
  imageryCheck: (id: string) =>
    api<Imagery>(`/api/runs/${encodeURIComponent(id)}/imagery/check`, { method: "POST" }),
  procure: (id: string, product: string, charge_to: string | null, note: string | null) =>
    api<ProcurementResponse>(`/api/runs/${encodeURIComponent(id)}/procurement`, {
      method: "POST",
      json: { product, charge_to, note },
    }),
  setSource: (id: string, key: string, mode: SourceMode, upload_id?: number, mapping?: Mapping) =>
    api<RunDetail>(`/api/runs/${encodeURIComponent(id)}/sources/${encodeURIComponent(key)}`, {
      method: "PUT",
      json: mode === "auto" ? { mode } : { mode, upload_id, mapping },
    }),
  mappingCheck: (uid: number, layer_key: string, mapping: Mapping, remember_as?: string | null, signal?: AbortSignal) =>
    api<MappingCheck>(`/api/uploads/${uid}/mapping/check`, {
      method: "POST",
      json: { layer_key, mapping, remember_as: remember_as || null },
      signal,
    }),
  setRule: (id: string, ruleId: string, body: { enabled: boolean; params: Record<string, unknown>; reason: string | null }) =>
    api<RulesResponse>(`/api/runs/${encodeURIComponent(id)}/rules/${encodeURIComponent(ruleId)}`, {
      method: "PUT",
      json: body,
    }),
  resetRules: (id: string) => api<RulesResponse>(`/api/runs/${encodeURIComponent(id)}/rules`, { method: "DELETE" }),
  setCosts: (id: string, body: Record<string, Record<string, number>>) =>
    api<CostsResponse>(`/api/runs/${encodeURIComponent(id)}/costs`, { method: "PUT", json: body }),
  extract: (id: string, layers?: string[], forceModels = false) =>
    api<Job>(`/api/runs/${encodeURIComponent(id)}/extract`, {
      method: "POST",
      json: { ...(layers && layers.length ? { layers } : {}), ...(forceModels ? { force_models: true } : {}) },
    }),
  imageryCatalog: (id: string) => api<ImageryCatalog>(`/api/runs/${encodeURIComponent(id)}/imagery/catalog`),
  fetchImagery: (id: string, products: string[]) =>
    api<Job>(`/api/runs/${encodeURIComponent(id)}/imagery/fetch`, { method: "POST", json: { products } }),
  models: () => api<{ models: ModelStatus[] }>("/api/models"),
  /** exports=false: quick test solve — routes only; the export set is made later on request. */
  solve: (id: string, exports = true) =>
    api<Job>(`/api/runs/${encodeURIComponent(id)}/solve`, { method: "POST", json: { exports } }),
  makeExports: (sid: number) => api<Solve>(`/api/solves/${sid}/exports`, { method: "POST" }),
  /** Re-pull a layer's online source (OSM power, Tarang Sanchar towers) for the run's area, update the store, re-extract it. */
  refreshLayer: (id: string, key: string) =>
    api<Job>(`/api/runs/${encodeURIComponent(id)}/layers/${encodeURIComponent(key)}/refresh`, { method: "POST" }),
  cancelJob: (jobId: number) => api<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
};

export function fileUrl(solveId: number, name: string) {
  return `/api/solves/${solveId}/files/${encodeURIComponent(name)}`;
}

export function useImageryCatalog(id: string | undefined, poll = false) {
  return useQuery({
    queryKey: ["imagery-catalog", id],
    queryFn: () => apiCalls.imageryCatalog(id as string),
    enabled: Boolean(id),
    staleTime: 10_000,
    refetchInterval: poll ? 5_000 : false,
  });
}

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: () => apiCalls.models(), staleTime: 60_000 });
}
