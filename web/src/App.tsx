import { lazy, Suspense } from "react";
import { Link, Navigate, Outlet, Route, Routes, useOutletContext, useParams } from "react-router-dom";
import Shell from "./components/Shell";
import { ErrorNote, Loading } from "./components/ui";
import { useRun } from "./api/hooks";
import type { RunDetail } from "./api/types";
import Home from "./screens/Home";
import Corridor from "./screens/Corridor";
import Images from "./screens/Images";
import Sources from "./screens/Sources";
import Rules from "./screens/Rules";
import Costs from "./screens/Costs";
import Extracting from "./screens/Extracting";
import Solve from "./screens/Solve";
import Why from "./screens/Why";
import Export from "./screens/Export";

const LayerReview = lazy(() => import("./screens/LayerReview"));
const Results = lazy(() => import("./screens/Results"));

export function useRunCtx(): RunDetail {
  return useOutletContext<RunDetail>();
}

function RunGate() {
  const { runId } = useParams();
  const q = useRun(runId);
  if (q.isPending) return <Loading text={`Opening ${runId}…`} />;
  if (q.isError)
    return (
      <div>
        <ErrorNote error={q.error} retry={() => void q.refetch()} />
        <Link to="/" className="btn ghost">
          Back to my routes
        </Link>
      </div>
    );
  return (
    <Suspense fallback={<Loading text="Loading the map…" />}>
      <Outlet context={q.data} />
    </Suspense>
  );
}

function RunIndex() {
  const run = useRunCtx();
  const to =
    run.job?.kind === "extract"
      ? "extracting"
      : run.job?.kind === "solve"
        ? "solve"
        : run.latest_solve_id
          ? "map"
          : run.status === "review"
            ? "layers"
            : "corridor";
  return <Navigate to={to} replace />;
}

function NotFound() {
  return (
    <div>
      <div className="head">
        <h2>Nothing here</h2>
        <p>That address does not match a screen.</p>
      </div>
      <Link to="/" className="btn">
        My routes
      </Link>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Home />} />
        <Route path="new" element={<Corridor />} />
        <Route path="runs/:runId" element={<RunGate />}>
          <Route index element={<RunIndex />} />
          <Route path="corridor" element={<Corridor />} />
          <Route path="images" element={<Images />} />
          <Route path="sources" element={<Sources />} />
          <Route path="rules" element={<Rules />} />
          <Route path="costs" element={<Costs />} />
          <Route path="review" element={<Navigate to="../extracting" replace />} />
          <Route path="extracting" element={<Extracting />} />
          <Route path="layers" element={<LayerReview />} />
          <Route path="solve" element={<Solve />} />
          <Route path="map" element={<Results />} />
          <Route path="why" element={<Why />} />
          <Route path="export" element={<Export />} />
          <Route path="*" element={<NotFound />} />
        </Route>
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
