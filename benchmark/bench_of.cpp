#include <algorithm>
#include <vector>

#include <benchmark/benchmark.h>

#include "xsimd_algorithm/stl/all_of.hpp"
#include "xsimd_algorithm/stl/any_of.hpp"
#include "xsimd_algorithm/stl/none_of.hpp"

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
    void BM_std_all_of(benchmark::State& state)
    {
        const auto constant = T { 5 };

        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        for (auto _ : state)
        {
            auto result = std::all_of(src.begin(), src.end(), [constant](const auto& x) { return x == constant; });
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
    
    template <typename T>
    void BM_xsimd_all_of(benchmark::State& state)
    {
        const auto constant = T { 5 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        for (auto _ : state)
        {
            auto result = xsimd::all_of(src.begin(), src.end(), [constant](const auto& x) { return x == constant; });
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_std_any_of(benchmark::State& state)
    {
        const auto constant = T { 5 };
        const auto tested = T { 10 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        for (auto _ : state)
        {
            auto result = std::any_of(src.begin(), src.end(), [tested](const auto& x) { return x == tested; });
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
    
    template <typename T>
    void BM_xsimd_any_of(benchmark::State& state)
    {
        const auto constant = T { 5 };
        const auto tested = T { 10 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        for (auto _ : state)
        {
            auto result = xsimd::any_of(src.begin(), src.end(), [tested](const auto& x) { return x == tested; });
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    static constexpr long long kSmall = 1LL << 12;
    static constexpr long long kMedium = 1LL << 18;
    static constexpr long long kLarge = 1LL << 22;

    // all_of
    BENCHMARK_TEMPLATE(BM_std_all_of, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_all_of, float)
    BENCH_ARGS;
    // any_of
    BENCHMARK_TEMPLATE(BM_std_any_of, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_any_of, float)
    BENCH_ARGS;

}