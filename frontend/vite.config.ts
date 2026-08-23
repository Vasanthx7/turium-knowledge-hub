import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dev server proxies /api to the FastAPI backend so the browser makes
// same-origin requests (no CORS friction in development). In production the
// frontend is served as static files and points at VITE_API_BASE_URL.
// The backend origin can be overridden with VITE_PROXY_TARGET (defaults to
// the local backend on :8000).
const proxyTarget = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: proxyTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
