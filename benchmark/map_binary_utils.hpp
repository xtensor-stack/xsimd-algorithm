/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>
#include <cstdint>
#include <format>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench_utils.hpp"
#include "xsimd_algorithm/map.hpp"
#include "xsimd_test_utils/utils.hpp"

namespace xsimd::bench
{
    using xsimd::alignment_options;
    using xsimd::map_options;

    template <typename Op, typename Alloc, typename Apply>
    void bench_binary(benchmark::State& state, Apply apply)
    {
        using lhs_t = typename Op::lhs_t;
        using rhs_t = typename Op::rhs_t;
        using output_t = typename Op::output_t;

        auto const size = static_cast<std::size_t>(state.range(0));
        auto [lhs, rhs, output] = Op::template make_input_output<Alloc>(size);

        for (auto _ : state)
        {
            apply(xsimd::test::as_span(lhs), xsimd::test::as_span(rhs), xsimd::test::as_span(output));
            benchmark::DoNotOptimize(output.data());
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
        state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations() * size * (sizeof(lhs_t) + sizeof(rhs_t) + sizeof(output_t))));
    }

    template <
        typename Op,
        typename Alloc,
        typename Arch,
        alignment_options aligned = alignment_options {},
        map_options opts = map_options {}>
    void bench_map_binary(benchmark::State& state)
    {
        bench_binary<Op, Alloc>(
            state,
            [](auto lhs, auto rhs, auto out)
            { Op::template range_apply_map_binary<aligned, opts, Arch>(lhs, rhs, out); });
    }

    template <typename Op, typename Alloc>
    void bench_binary_scalar(benchmark::State& state)
    {
        bench_binary<Op, Alloc>(
            state,
            [](auto lhs, auto rhs, auto out)
            { Op::range_apply_scalar(lhs, rhs, out); });
    }

    template <typename Op, typename Arch, typename Bench>
    void register_binary_bench(
        std::string_view variant,
        Bench bench_fn,
        std::vector<std::int64_t> const& sizes = xsimd::bench::bench_sizes<typename Op::lhs_t>())
    {
        using lhs_t = typename Op::lhs_t;
        using rhs_t = typename Op::rhs_t;

        auto* bench = benchmark::RegisterBenchmark(
            std::format(
                "{}/{}/{}_{}/{}",
                Arch::name(),
                Op::name,
                xsimd::bench::type_name<lhs_t>(),
                xsimd::bench::type_name<rhs_t>(),
                variant),
            bench_fn);
        for (auto const size : sizes)
        {
            bench->Arg(size);
        }
    }
}
