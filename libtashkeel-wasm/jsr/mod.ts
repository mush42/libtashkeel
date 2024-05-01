// deno-lint-ignore-file
// deno-fmt-ignore-file

import { do_tashkeel, instantiate } from "./libtashkeel_wasm.generated.js";

await instantiate();

export default do_tashkeel;