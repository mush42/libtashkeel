use std::env;

fn main() {
    println!("cargo:rerun-if-changed=./src/lib.rs");

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_include_version(true)
        .with_documentation(false)
        .with_parse_deps(true)
        .with_parse_include(&["ffi-support"])
        .with_cpp_compat(true)
        // .with_language(cbindgen::Language::Cxx)
        .include_item("ExternError")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("libtashkeel.h");
}
