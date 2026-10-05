import { useEffect, useRef, type ReactNode } from "react";
import type { RunStatus, SourceMode } from "../api/types";
import { errorText } from "../api/client";

export function Spinner({ label }: { label?: string }) {
  return <span className="spin" role="status" aria-label={label ?? "Loading"} />;
}

export function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <div className="loading">
      <Spinner /> {text}
    </div>
  );
}

/** Inline error note with the server's detail text, and an optional retry. */
export function ErrorNote({ error, retry, title }: { error: unknown; retry?: () => void; title?: string }) {
  if (!error) return null;
  return (
    <div className="note n-stop" role="alert" style={{ margin: "10px 0" }}>
      <div style={{ flex: 1 }}>
        {title ? <b>{title} </b> : null}
        {errorText(error)}
      </div>
      {retry ? (
        <button className="mini" onClick={retry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function Note({ kind, children, style }: { kind: "warn" | "stop" | "info" | "ok"; children: ReactNode; style?: React.CSSProperties }) {
  return (
    <div className={`note n-${kind}`} style={style}>
      <div style={{ minWidth: 0 }}>{children}</div>
    </div>
  );
}

/** A button that shows a spinner and is disabled while its mutation is pending. */
export function Btn({
  pending,
  children,
  ghost,
  wide,
  disabled,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { pending?: boolean; ghost?: boolean; wide?: boolean }) {
  return (
    <button
      {...rest}
      className={`btn${ghost ? " ghost" : ""}${wide ? " wide" : ""}${className ? ` ${className}` : ""}`}
      disabled={disabled || pending}
      aria-busy={pending || undefined}
    >
      {pending ? <span className="spin" aria-hidden /> : null}
      {children}
    </button>
  );
}

const STATUS: Record<RunStatus, [string, string]> = {
  draft: ["p-lock", "Draft"],
  extracting: ["p-acc", "Extracting"],
  review: ["p-cross", "Layers ready to review"],
  solving: ["p-acc", "Solving"],
  complete: ["p-go", "Complete"],
  blocked: ["p-nogo", "No viable route"],
  failed: ["p-nogo", "Failed"],
};

export function StatusPill({ status, extra }: { status: RunStatus; extra?: string | null }) {
  const [cls, txt] = STATUS[status] ?? ["p-lock", status];
  return (
    <span className={`pill ${cls}`}>
      {txt}
      {extra ? ` · ${extra}` : ""}
    </span>
  );
}

export function SourcePill({ mode, producer }: { mode?: SourceMode | string | null; producer?: string | null }) {
  if (mode === "replace") return <span className="pill p-chg">your file</span>;
  if (mode === "add") return <span className="pill p-acc">supplemented</span>;
  if (producer === "reference") return <span className="pill p-go">cached</span>;
  return <span className="pill p-go">auto</span>;
}

export function ConfPill({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return <span className="hint">n/a</span>;
  return <span className={`pill ${value < 0.8 ? "p-nogo" : "p-go"}`}>{value.toFixed(2)}</span>;
}

export function BestGuess() {
  return (
    <i className="bg" title="Best guess — not yet confirmed by a source document">
      best guess
    </i>
  );
}

/** Native <dialog> driven by an `open` prop. Escape / backdrop close call onClose. */
export function Dialog({
  open,
  onClose,
  id,
  children,
  labelledBy,
}: {
  open: boolean;
  onClose: () => void;
  id?: string;
  children: ReactNode;
  labelledBy?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      try {
        d.showModal();
      } catch {
        d.setAttribute("open", "");
      }
    } else if (!open && d.open) {
      d.close();
    }
  }, [open]);
  return (
    <dialog
      ref={ref}
      id={id}
      aria-labelledby={labelledBy}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      {open ? <div className="dlg">{children}</div> : null}
    </dialog>
  );
}

export function Prog({ value }: { value: number | null | undefined }) {
  const w = Math.max(0, Math.min(100, Math.round((value ?? 0) * 100)));
  return (
    <div className="prog" role="progressbar" aria-valuenow={w} aria-valuemin={0} aria-valuemax={100}>
      <i style={{ width: `${w}%` }} />
    </div>
  );
}

export function Kpi({ k, v, unit, color }: { k: string; v: ReactNode; unit?: string; color?: string }) {
  return (
    <div className="kpi">
      <div className="k">{k}</div>
      <div className="v" style={color ? { color } : undefined}>
        {v}
        {unit ? <small> {unit}</small> : null}
      </div>
    </div>
  );
}

export function Head({ title, children, right }: { title: string; children?: ReactNode; right?: ReactNode }) {
  if (right) {
    return (
      <div className="head row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h2>{title}</h2>
          {children ? <p>{children}</p> : null}
        </div>
        {right}
      </div>
    );
  }
  return (
    <div className="head">
      <h2>{title}</h2>
      {children ? <p>{children}</p> : null}
    </div>
  );
}
