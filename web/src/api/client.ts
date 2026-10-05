import type { Job } from "./types";

/** An HTTP error from the Beeline API, carrying the server's `detail` text. */
export class ApiError extends Error {
  status: number;
  detail: string;
  busy: boolean;
  job: Job | null;

  constructor(status: number, detail: string, busy = false, job: Job | null = null) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.busy = busy;
    this.job = job;
  }
}

/** Turn a FastAPI `detail` (string, or a 422 list of {loc,msg}) into one readable sentence. */
export function detailText(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      if (d && typeof d === "object" && "msg" in d) {
        const loc = Array.isArray((d as { loc?: unknown[] }).loc)
          ? ((d as { loc: unknown[] }).loc.filter((x) => x !== "body").join("."))
          : "";
        return loc ? `${loc}: ${String((d as { msg: unknown }).msg)}` : String((d as { msg: unknown }).msg);
      }
      return String(d);
    });
    if (parts.length) return parts.join("; ");
  }
  return fallback;
}

export function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  if (e instanceof DOMException && e.name === "AbortError") return "The request was cancelled.";
  if (e instanceof TypeError) return "Could not reach the Beeline server. Check that it is running and try again.";
  if (e instanceof Error) return e.message;
  return String(e);
}

async function parse(res: Response): Promise<unknown> {
  const ct = res.headers.get("content-type") ?? "";
  if (res.status === 204) return null;
  if (ct.includes("application/json")) return res.json();
  const t = await res.text();
  try {
    return JSON.parse(t);
  } catch {
    return t;
  }
}

export async function api<T>(path: string, init?: RequestInit & { json?: unknown }): Promise<T> {
  const { json, ...rest } = init ?? {};
  const headers = new Headers(rest.headers);
  let body = rest.body;
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  }
  headers.set("Accept", "application/json");
  const res = await fetch(path, { ...rest, headers, body });
  const data = await parse(res);
  if (!res.ok) {
    const obj = (data && typeof data === "object" ? data : {}) as { detail?: unknown; busy?: boolean; job?: Job };
    const fallback =
      res.status >= 500
        ? `The server failed (HTTP ${res.status}). Try again; if it keeps failing, check the API log.`
        : `Request failed (HTTP ${res.status}).`;
    const text = typeof data === "string" && data && res.status < 500 && data.length < 300 ? data : detailText(obj.detail, fallback);
    throw new ApiError(res.status, text, Boolean(obj.busy), obj.job ?? null);
  }
  return data as T;
}

/** Multipart upload with progress (fetch has no upload progress events). */
export function uploadFile<T>(
  path: string,
  file: File,
  onProgress: (frac: number) => void,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.responseType = "text";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let data: unknown = xhr.responseText;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data as T);
      } else {
        const obj = (data && typeof data === "object" ? data : {}) as { detail?: unknown };
        reject(new ApiError(xhr.status, detailText(obj.detail, `Upload failed (HTTP ${xhr.status}).`)));
      }
    };
    xhr.onerror = () => reject(new TypeError("network"));
    xhr.onabort = () => reject(new DOMException("aborted", "AbortError"));
    if (signal) signal.addEventListener("abort", () => xhr.abort(), { once: true });
    const fd = new FormData();
    fd.append("file", file, file.name);
    xhr.send(fd);
  });
}
