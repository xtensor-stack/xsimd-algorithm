#include <algorithm>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>

#include "xsimd_algorithm/stl/arange.hpp"
#include "xsimd_algorithm/stl/reduce.hpp"
#include "xsimd_algorithm/stl/transform.hpp"

#define BENCH_ARGS ->Arg(kSmall)->Arg(kMedium)->Arg(kLarge)
namespace
{
    template <typename T>
    std::vector<T> make_input(std::size_t n)
    {
        std::vector<T> v(n);
        std::iota(v.begin(), v.end(), T(1));
        return v;
    }

    template <typename T>
    void BM_std_arange(benchmark::State& state)
    {
        std::vector<T> v(static_cast<std::size_t>(state.range(0)));
        for (auto _ : state)
        {
            std::iota(v.begin(), v.end(), T(0));
            benchmark::DoNotOptimize(v.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_xsimd_arange(benchmark::State& state)
    {
        std::vector<T> v(static_cast<std::size_t>(state.range(0)));
        for (auto _ : state)
        {
            xsimd::arange(v.begin(), v.end(), T(0), T(1));
            benchmark::DoNotOptimize(v.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_std_transform_unary(benchmark::State& state)
    {
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)));
        std::vector<T> dst(src.size());
        for (auto _ : state)
        {
            std::transform(src.begin(), src.end(), dst.begin(),
                           [](T x) noexcept
                           { return x * x; });
            benchmark::DoNotOptimize(dst.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_xsimd_transform_unary(benchmark::State& state)
    {
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)));
        std::vector<T> dst(src.size());
        for (auto _ : state)
        {
            xsimd::transform(src.begin(), src.end(), dst.begin(),
                             [](auto x) noexcept
                             { return x * x; });
            benchmark::DoNotOptimize(dst.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_std_transform_binary(benchmark::State& state)
    {
        std::size_t n = static_cast<std::size_t>(state.range(0));
        auto a = make_input<T>(n);
        auto b = make_input<T>(n);
        std::vector<T> dst(n);
        for (auto _ : state)
        {
            std::transform(a.begin(), a.end(), b.begin(), dst.begin(),
                           [](T x, T y) noexcept
                           { return x + y; });
            benchmark::DoNotOptimize(dst.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_xsimd_transform_binary(benchmark::State& state)
    {
        std::size_t n = static_cast<std::size_t>(state.range(0));
        auto a = make_input<T>(n);
        auto b = make_input<T>(n);
        std::vector<T> dst(n);
        for (auto _ : state)
        {
            xsimd::transform(a.begin(), a.end(), b.begin(), dst.begin(),
                             [](auto x, auto y) noexcept
                             { return x + y; });
            benchmark::DoNotOptimize(dst.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_std_reduce(benchmark::State& state)
    {
        auto v = make_input<T>(static_cast<std::size_t>(state.range(0)));
        for (auto _ : state)
        {
            T result = std::reduce(v.begin(), v.end(), T(0));
            benchmark::DoNotOptimize(result);
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_xsimd_reduce(benchmark::State& state)
    {
        auto v = make_input<T>(static_cast<std::size_t>(state.range(0)));
        for (auto _ : state)
        {
            T result = xsimd::reduce(v.begin(), v.end(), T(0));
            benchmark::DoNotOptimize(result);
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    static constexpr long long kSmall = 1LL << 12;
    static constexpr long long kMedium = 1LL << 18;
    static constexpr long long kLarge = 1LL << 22;

    // arange (std::iota is the scalar equivalent of xsimd::arange)
    BENCHMARK_TEMPLATE(BM_std_arange, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_arange, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_std_arange, double)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_arange, double)
    BENCH_ARGS;

    // transform unary
    BENCHMARK_TEMPLATE(BM_std_transform_unary, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_transform_unary, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_std_transform_unary, double)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_transform_unary, double)
    BENCH_ARGS;

    // transform binary
    BENCHMARK_TEMPLATE(BM_std_transform_binary, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_transform_binary, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_std_transform_binary, double)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_transform_binary, double)
    BENCH_ARGS;

    // reduce
    BENCHMARK_TEMPLATE(BM_std_reduce, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_reduce, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_std_reduce, double)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_reduce, double)
    BENCH_ARGS;

}

BENCHMARK_MAIN();
