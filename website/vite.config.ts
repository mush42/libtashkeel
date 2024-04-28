import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';
import topLevelAwait from "vite-plugin-top-level-await";
import wasmPlugin from 'vite-plugin-wasm';

export default defineConfig({
  base: "/libtashkeel",
  plugins: [
    preact(),
    topLevelAwait(),
    wasmPlugin(),
  ],
});
