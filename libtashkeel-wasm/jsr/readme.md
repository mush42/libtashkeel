# Libtashkeel

`Libtashkeel`is a cross-platform library for diacritic restoration of Arabic text.

`Libtashkeel` is written in Rust, and provides both a **standalone** linkable library and a command line tool.

The library uses models trained mainly on MSA data, from [Hareef](https://github.com/mush42/hareef).

## Getting Libtashkeel

```bash
deno add nlp/tashkeel@1.6.3
```

## Usage 

```js
import do_tashkeel from "tashkeel";
const tashkeeled = tashkeel("مرحبا بك");
```

# Licence

Copyright (c) Musharraf Omer. This project is licenced under the terms of The MIT License

