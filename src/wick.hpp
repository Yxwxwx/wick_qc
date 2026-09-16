#pragma once

// Public umbrella. Narrower module headers remain available for numeric-only
// consumers and build-time generated kernels.
#include "backend/ndarray.hpp" // IWYU pragma: export
#include "codegen/cpp_emitter.hpp" // IWYU pragma: export
#include "einsum/einsum.hpp" // IWYU pragma: export
#include "equation/graph.hpp" // IWYU pragma: export
#include "method/nevpt2.hpp" // IWYU pragma: export
#include "method/rhf.hpp" // IWYU pragma: export
#include "method/spatial.hpp" // IWYU pragma: export
#include "method/specification.hpp" // IWYU pragma: export
#include "method/spin_orbital.hpp" // IWYU pragma: export
#include "runtime/executor.hpp" // IWYU pragma: export
#include "runtime/numeric.hpp" // IWYU pragma: export
#include "runtime/spatial.hpp" // IWYU pragma: export
#include "symbolic/index_domain.hpp" // IWYU pragma: export
#include "symbolic/wick.hpp" // IWYU pragma: export

// LAPACK is optional for consumers that only use Wick algebra and einsum.
#if defined(WICKQC_LAPACK_MKL) || defined(WICKQC_LAPACK_OPENBLAS) || \
    defined(WICKQC_LAPACK_NETLIB) || defined(WICKQC_LAPACK_EIGEN)
#include "backend/lapack.hpp" // IWYU pragma: export
#endif
