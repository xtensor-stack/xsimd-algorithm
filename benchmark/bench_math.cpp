/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>

#include <benchmark/benchmark.h>

#include "map_binary_utils.hpp"
#include "map_unary_utils.hpp"
#include "xsimd_test_utils/map_binary_data.hpp"
#include "xsimd_test_utils/map_unary_data.hpp"
#include "xsimd_test_utils/utils.hpp"

namespace
{
    using xsimd::bench::bench_binary_scalar;
    using xsimd::bench::bench_map_binary;
    using xsimd::bench::bench_map_unary;
    using xsimd::bench::bench_scalar;
    using xsimd::bench::register_binary_bench;
    using xsimd::bench::register_bench;
    using xsimd::alignment_options;

    /// Register math benchmarks.
    ///
    /// To avoid an explosion of benchmarks, we only add a simple aligned benchmark.
    /// This will let us know the performance of the xsimd wrappers.
    /// See bench_map for benchmarks on the different flavor of mapping, alignment,
    /// headers and trailers.
    ///
    /// This benchmark aims to test raw the performance of intrinsic, unrelated to
    /// how they are iterated on (alignment, memory etc). To do so, they aim to stay
    /// in L1 cache.
    template <typename Op>
    void register_benches()
    {
        using input_t = typename Op::input_t;
        using arch = xsimd::default_arch;
        using aligned_alloc = typename xsimd::test::aligned_vector<input_t, arch>::allocator_type;

        register_bench<Op, arch>(
            "hot/scalar", bench_scalar<Op, aligned_alloc>, /* sizes = */ { 1024 });
        register_bench<Op, arch>(
            "hot/simd",
            bench_map_unary<Op, aligned_alloc, arch, alignment_options { .start_aligned = true, .end_aligned = true }>,
            /* sizes = */ { 1024 });
    }

    template <typename Op>
    void register_binary_benches()
    {
        using lhs_t = typename Op::lhs_t;
        using arch = xsimd::default_arch;
        using aligned_alloc = typename xsimd::test::aligned_vector<lhs_t, arch>::allocator_type;

        register_binary_bench<Op, arch>(
            "hot/scalar", bench_binary_scalar<Op, aligned_alloc>, /* sizes = */ { 1024 });
        register_binary_bench<Op, arch>(
            "hot/simd",
            bench_map_binary<Op, aligned_alloc, arch, alignment_options { .start_aligned = true, .end_aligned = true }>,
            /* sizes = */ { 1024 });
    }

    bool const registered = []
    {
        register_benches<xsimd::test::sqrt_op<float>>();
        register_benches<xsimd::test::sqrt_op<double>>();
        register_benches<xsimd::test::abs_op<float>>();
        register_benches<xsimd::test::abs_op<double>>();
        register_benches<xsimd::test::exp_op<float>>();
        register_benches<xsimd::test::exp_op<double>>();
        register_benches<xsimd::test::widen_op<std::int8_t>>();
        register_benches<xsimd::test::widen_op<std::int16_t>>();
        register_benches<xsimd::test::widen_op<std::int32_t>>();
        register_binary_benches<xsimd::test::add_op<std::int32_t>>();
        register_binary_benches<xsimd::test::multiply_op<float>>();
        return true;
    }();
}
