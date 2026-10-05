import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";
import App from "./App";
import { queryClient } from "./api/hooks";

// Never let a stray rejection surface as an unhandled error; screens show their own errors.
window.addEventListener("unhandledrejection", (e) => {
  if (e.reason && (e.reason.name === "AbortError" || e.reason.name === "ApiError")) e.preventDefault();
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
