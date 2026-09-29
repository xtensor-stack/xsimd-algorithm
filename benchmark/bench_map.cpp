/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>
#include <cstdint>

#include <benchmark/benchmark.h>

#include "map_binary_utils.hpp"
#include "map_unary_utils.hpp"
#include "xsimd_algorithm/map.hpp"
#include "xsimd_test_utils/map_binary_data.hpp"
#include "xsimd_test_utils/map_unary_data.hpp"
#include "xsimd_test_utils/utils.hpp"

namespace
{
    using xsimd::bench::bench_binary_scalar;
    using xsimd::bench::bench_map_binary;
    using xsimd::bench::bench_map_unary;
    using xsimd::bench::bench_scalar;
    using xsimd::bench::bench_transform;
    using xsimd::bench::register_binary_bench;
    using xsimd::bench::register_bench;
    using xsimd::alignment_options;
    using xsimd::map_options;

    template <typename Op>
    void register_benches()
    {
        using input_t = typename Op::input_t;
        using arch = xsimd::default_arch;
        using aligned_alloc = typename xsimd::test::aligned_vector<input_t, arch>::allocator_type;
        using unaligned_alloc = typename xsimd::test::unaligned_vector<input_t, arch>::allocator_type;

        constexpr auto noalign = alignment_options {};
        constexpr auto noopts = map_options { .unroll_factor = 1, .pure = false };

        register_bench<Op, arch>("aligned/scalar", bench_scalar<Op, aligned_alloc>);
        register_bench<Op, arch>("aligned/simd/transform", bench_transform<Op, aligned_alloc, arch>);
        register_bench<Op, arch>(
            "aligned/simd/map",
            bench_map_unary<Op, aligned_alloc, arch, noalign, noopts>);
        register_bench<Op, arch>(
            "aligned/simd/map:pure",
            bench_map_unary<
                Op, aligned_alloc, arch,
                noalign, map_options { .unroll_factor = 1, .pure = true }>);
        register_bench<Op, arch>(
            "aligned/simd/map:unroll4",
            bench_map_unary<
                Op, aligned_alloc, arch, noalign, map_options { .unroll_factor = 4 }>);
        register_bench<Op, arch>(
            "aligned/simd/map:noheader",
            bench_map_unary<
                Op, aligned_alloc, arch,
                alignment_options { .start_aligned = true }, noopts>);
        register_bench<Op, arch>(
            "aligned/simd/map:noheader+pure+unroll4",
            bench_map_unary<
                Op, aligned_alloc, arch,
                alignment_options { .start_aligned = true }, map_options { .unroll_factor = 4, .pure = true }>);

        register_bench<Op, arch>("unaligned/scalar", bench_scalar<Op, unaligned_alloc>);
        register_bench<Op, arch>("unaligned/simd/transform", bench_transform<Op, unaligned_alloc, arch>);
        register_bench<Op, arch>(
            "unaligned/simd/map",
            bench_map_unary<Op, unaligned_alloc, arch, noalign, noopts>);
        register_bench<Op, arch>(
            "unaligned/simd/map:pure",
            bench_map_unary<
                Op, unaligned_alloc, arch,
                noalign, map_options { .unroll_factor = 1, .pure = true }>);
        register_bench<Op, arch>(
            "unaligned/simd/map:unroll4",
            bench_map_unary<
                Op, unaligned_alloc, arch, noalign, map_options { .unroll_factor = 4 }>);
        register_bench<Op, arch>(
            "unaligned/simd/map:pure+unroll4",
            bench_map_unary<
                Op, unaligned_alloc, arch,
                noalign, map_options { .unroll_factor = 4, .pure = true }>);
    }

    template <typename Op>
    void register_binary_benches()
    {
        using lhs_t = typename Op::lhs_t;
        using arch = xsimd::default_arch;
        using aligned_alloc = typename xsimd::test::aligned_vector<lhs_t, arch>::allocator_type;
        using unaligned_alloc = typename xsimd::test::unaligned_vector<lhs_t, arch>::allocator_type;

        constexpr auto noalign = alignment_options {};
        constexpr auto noopts = map_options { .unroll_factor = 1, .pure = false };

        register_binary_bench<Op, arch>("aligned/scalar", bench_binary_scalar<Op, aligned_alloc>);
        register_binary_bench<Op, arch>(
            "aligned/simd/map",
            bench_map_binary<Op, aligned_alloc, arch, noalign, noopts>);
        register_binary_bench<Op, arch>(
            "aligned/simd/map:pure",
            bench_map_binary<
                Op, aligned_alloc, arch,
                noalign, map_options { .unroll_factor = 1, .pure = true }>);
        register_binary_bench<Op, arch>(
            "aligned/simd/map:unroll4",
            bench_map_binary<
                Op, aligned_alloc, arch, noalign, map_options { .unroll_factor = 4 }>);
        register_binary_bench<Op, arch>(
            "aligned/simd/map:noheader",
            bench_map_binary<
                Op, aligned_alloc, arch,
                alignment_options { .start_aligned = true }, noopts>);
        register_binary_bench<Op, arch>(
            "aligned/simd/map:noheader+pure+unroll4",
            bench_map_binary<
                Op, aligned_alloc, arch,
                alignment_options { .start_aligned = true }, map_options { .unroll_factor = 4, .pure = true }>);

        register_binary_bench<Op, arch>("unaligned/scalar", bench_binary_scalar<Op, unaligned_alloc>);
        register_binary_bench<Op, arch>(
            "unaligned/simd/map",
            bench_map_binary<Op, unaligned_alloc, arch, noalign, noopts>);
        register_binary_bench<Op, arch>(
            "unaligned/simd/map:pure",
            bench_map_binary<
                Op, unaligned_alloc, arch,
                noalign, map_options { .unroll_factor = 1, .pure = true }>);
        register_binary_bench<Op, arch>(
            "unaligned/simd/map:unroll4",
            bench_map_binary<
                Op, unaligned_alloc, arch, noalign, map_options { .unroll_factor = 4 }>);
        register_binary_bench<Op, arch>(
            "unaligned/simd/map:pure+unroll4",
            bench_map_binary<
                Op, unaligned_alloc, arch,
                noalign, map_options { .unroll_factor = 4, .pure = true }>);
    }

    bool const registered = []
    {
        register_benches<xsimd::test::sqrt_op<float>>();
        register_benches<xsimd::test::sqrt_op<double>>();
        register_benches<xsimd::test::abs_op<float>>();
        register_benches<xsimd::test::abs_op<double>>();
        register_benches<xsimd::test::exp_op<float>>();
        register_benches<xsimd::test::exp_op<double>>();
        register_binary_benches<xsimd::test::add_op<std::int32_t>>();
        register_binary_benches<xsimd::test::multiply_op<float>>();
        register_binary_benches<xsimd::test::mixed_multiply_op>();
        return true;
    }();
}
