import { do_tashkeel, instantiate } from "./lib/libtashkeel_wasm.generated.js";

if (import.meta.main) {
  await instantiate();
  console.log(do_tashkeel("بسم الله الرحمن الرحيم"));
}
