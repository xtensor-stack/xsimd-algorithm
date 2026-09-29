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
    void bench_unary(benchmark::State& state, Apply apply)
    {
        using input_t = typename Op::input_t;

        auto const size = static_cast<std::size_t>(state.range(0));
        auto [input, output] = Op::template make_input_output<Alloc>(size);

        for (auto _ : state)
        {
            apply(xsimd::test::as_span(input), xsimd::test::as_span(output));
            benchmark::DoNotOptimize(output.data());
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
        state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations() * size * 2 * sizeof(input_t)));
    }

    template <
        typename Op,
        typename Alloc,
        typename Arch,
        alignment_options aligned = alignment_options {},
        map_options opts = map_options {}>
    void bench_map_unary(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(
            state,
            [](auto in, auto out)
            { Op::template range_apply_map_unary<aligned, opts, Arch>(in, out); });
    }

    template <typename Op, typename Alloc, typename Arch>
    void bench_transform(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(
            state,
            [](auto in, auto out)
            { Op::template range_apply_transform<Arch>(in, out); });
    }

    template <typename Op, typename Alloc>
    void bench_scalar(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(state, [](auto in, auto out)
                               { Op::range_apply_scalar(in, out); });
    }

    template <typename Op, typename Arch, typename Bench>
    void register_bench(
        std::string_view variant,
        Bench bench_fn,
        std::vector<std::int64_t> const& sizes = xsimd::bench::bench_sizes<typename Op::input_t>())
    {
        using input_t = typename Op::input_t;

        auto* bench = benchmark::RegisterBenchmark(
            std::format("{}/{}/{}/{}", Arch::name(), Op::name, xsimd::bench::type_name<input_t>(), variant),
            bench_fn);
        for (auto const size : sizes)
        {
            bench->Arg(size);
        }
    }
}
