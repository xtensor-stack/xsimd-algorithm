#include <algorithm>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>

#include "xsimd_algorithm/stl/find.hpp"

#define BENCH_ARGS ->Arg(kSmall)->Arg(kMedium)->Arg(kLarge)

namespace
{
    template <typename T>
    std::vector<T> make_input(size_t size, size_t position, T constant)
    {
        std::vector<T> result(size);
        std::iota(result.begin(), result.end(), T(1));
        if (position < size)
        {
            result[position] = constant;
        }
        return result;
    }

    template <typename T>
    void BM_std_find(benchmark::State& state)
    {
        const auto constant = T { -42 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(0)) - 1, constant);
        for (auto _ : state)
        {
            auto result = std::find(src.begin(), src.end(), constant);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
    
    template <typename T>
    void BM_xsimd_find(benchmark::State& state)
    {
        const auto constant = T { -42 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(0)) - 1, constant);
        for (auto _ : state)
        {
            auto result = xsimd::find(src.begin(), src.end(), constant);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    static constexpr long long kSmall = 1LL << 12;
    static constexpr long long kMedium = 1LL << 18;
    static constexpr long long kLarge = 1LL << 22;

    // find
    BENCHMARK_TEMPLATE(BM_std_find, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_find, float)
    BENCH_ARGS;

}