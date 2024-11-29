#pragma once

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = at::Half;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = at::BFloat16; \
      return __VA_ARGS__();                  \
    }                                        \
  }()


