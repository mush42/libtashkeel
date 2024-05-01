import do_tashkeel from './mod.ts';

if (import.meta.main) {
  const args = Deno.args;
  if (args.length !== 1) {
    console.error("Error: Please provide a single string argument.");
    Deno.exit(1); // Exit with an error code
  }

  const inputString = args[0];
  console.log(do_tashkeel(inputString));
}
