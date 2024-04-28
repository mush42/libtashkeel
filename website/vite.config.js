import { defineConfig } from 'vite';
import wasmPlugin from 'vite-plugin-wasm';
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  plugins: [
    wasmPlugin(),
    topLevelAwait(),
  ],
});
