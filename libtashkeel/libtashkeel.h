/* Generated with cbindgen:0.26.0 */

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

constexpr static const int32_t INPUT_TOO_LONG = 1;

constexpr static const int32_t INFERENCE_ERROR = 2;

constexpr static const int32_t MODEL_LOAD_ERROR = 3;

constexpr static const int32_t UNKNOWN_ERROR = 99;

using FfiStr = const char*;

using ErrorCode = int32_t;
constexpr static const ErrorCode ErrorCode_SUCCESS = 0;
constexpr static const ErrorCode ErrorCode_PANIC = -1;
constexpr static const ErrorCode ErrorCode_INVALID_HANDLE = -1000;

struct ExternError {
  ErrorCode code;
  char *message;
};

extern "C" {

char *libtashkeelTashkeel(FfiStr text_ptr, const float *taskeen_threshold, ExternError *out_error);

void libtashkeel_init(FfiStr model_path_ptr, ExternError *out_error);

} // extern "C"
