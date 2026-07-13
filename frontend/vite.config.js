import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",  // bind IPv4 explicitly; ::1-only binding breaks curl/proxy checks
    port: 5174,
    proxy: {
      // 127.0.0.1, not localhost: on Windows localhost can resolve to ::1
      // while Django listens on IPv4, silently breaking the proxy.
      "/api": { target: process.env.BACKEND_URL || "http://127.0.0.1:8000" },
    },
  },
});
