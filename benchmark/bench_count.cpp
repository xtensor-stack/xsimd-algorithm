#include <algorithm>
#include <vector>

#include <benchmark/benchmark.h>

#include "xsimd_algorithm/stl/count.hpp"

#define BENCH_ARGS ->Arg(kSmall)->Arg(kMedium)->Arg(kLarge)

namespace
{
    template <typename T>
    std::vector<T> make_input(std::size_t n, T constant)
    {
        std::vector<T> v(n);
        std::fill(v.begin(), v.end(), constant);
        return v;
    }

    template <typename T>
    void BM_std_count(benchmark::State& state)
    {
        const auto constant = T { 5 };

        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        
        for (auto _ : state)
        {
            auto result = std::count(src.begin(), src.end(), constant);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
    
    template <typename T>
    void BM_xsimd_count(benchmark::State& state)
    {
        const auto constant = T { 5 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        for (auto _ : state)
        {
            auto result = xsimd::count(src.begin(), src.end(), constant);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }


    static constexpr long long kSmall = 1LL << 12;
    static constexpr long long kMedium = 1LL << 18;
    static constexpr long long kLarge = 1LL << 22;

    // count
    BENCHMARK_TEMPLATE(BM_std_count, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_count, float)
    BENCH_ARGS;

}