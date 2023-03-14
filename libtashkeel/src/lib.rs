use ffi_support::{
    call_with_output, define_string_destructor, rust_string_to_c, ExternError, FfiStr,
};
use libc::c_char;
use libtashkeel_base::do_tashkeel;

define_string_destructor!(libtashkeel_free_string);

#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn libtashkeelTashkeel(
    text_ptr: FfiStr,
    out_error: &mut ExternError,
) -> *mut c_char {
    let text = text_ptr.into_string();
    call_with_output(out_error, move || rust_string_to_c(do_tashkeel(text)))
}
