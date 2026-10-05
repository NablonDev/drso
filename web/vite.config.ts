import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API = process.env.BEELINE_API ?? "http://localhost:8710";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: { "/api": { target: API, changeOrigin: true } },
  },
  preview: { port: 4173, proxy: { "/api": { target: API, changeOrigin: true } } },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks: { maplibre: ["maplibre-gl"], react: ["react", "react-dom", "react-router-dom"] },
      },
    },
  },
});
