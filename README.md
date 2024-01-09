# Libtashkeel

`Libtashkeel`is a cross-platform library for diacritic restoration of Arabic text.

`Libtashkeel` is written in Rust, and provides both a **standalone** linkable library and a command line tool.

The library uses models trained mainly on MSA data, from [Hareef](https://github.com/mush42/hareef).

## Getting Libtashkeel

You need to build the project yourself, see the **Building** section for a step-by-step guide.

## Usage

### Using the library

To use `Libtashkeel` from your C/C++ project, just include [libtashkeel.h](./libtashkeel/libtashkeel.h) and you are good to go.

The API consists of a single entry point for diacritizing a **utf-8 ** encoded string. Please take a look at [ffi_usage_example.py](./ffi_usage_example.py) for sample usage.

### From Python

**Python** bindings are also provided.

After building the wheels (see the **Building** section), install the wheel using `pip`:

```bash
pip install ./target/wheels/pylibtashkeel*.whl
```

and then:

```python
>>> from pylibtashkeel import tashkeel
>>> tashkeel("إن روعة اللغة العربية لا تتبدى إلا لعشاقها")
'إِنَّ رَوْعَةَ اللُّغَةِ الْعَرَبِيَّةِ لَا تَتَبَدَّى إِلَّا لِعُشَّاقِهَا'
```

### Command line tool

`Libtashkeel` provides a standalone executable called **tashkeel** for diacritizing text from the command line.

```bash
$ tashkeel --help
Arabic-text diacritic restoration using neural networks

Usage: tashkeel [OPTIONS]

Options:
  -f, --input-file <INPUT_FILE>    Input file (default `stdin`)
  -o, --output-file <OUTPUT_FILE>  Output file (default `stdout`)
  -i, --interactive                Use interactive mode (useful for testing)
  -t, --taskeen                    Use sukoon for case-ending diacritic if the model is uncertain
  -p, --prob <PROB>                Taskeen threshold probability [default: 0.95]
  -x, --onnx <ONNX_MODEL>          ONNX model (default: use bundled model if available)
  -h, --help                       Print help
  -V, --version                    Print version

```

## Building

`Libtashkeel` is written in **Rust**, [you need to install Rust first](https://www.rust-lang.org/tools/install)

To build the linkable library `libtashkeel`, and the command line tool `tashkeel`, run the following command from the root of the repository:

```bash
$ cargo build --release
```

Then, the built library and executable is found under `target` directory.

To build **Python** bindings as a wheel, you need to install [maturin](https://github.com/pyo3/maturin)

Run the following to build the wheel:

```bash
$ cd pylibtashkeel
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install maturin
$ maturin build --release --strip -i .venv/bin/python
```

Then, the built wheel is found under `target/wheels` directory.

# Licence

Copyright (c) Musharraf Omer. This project is licenced under the terms of The MIT License

