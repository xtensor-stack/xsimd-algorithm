#include <algorithm>
#include <execution>
#include <vector>

#include <benchmark/benchmark.h>

#include "xsimd_algorithm/stl/max_element.hpp"
#include "xsimd_algorithm/stl/min_element.hpp"

#define BENCH_ARGS ->Arg(kSmall)->Arg(kMedium)->Arg(kLarge)

namespace
{
     template <class ForwardIterator>
    typename std::iterator_traits<ForwardIterator>::value_type dummy_std_max_value(ForwardIterator first, ForwardIterator last) noexcept
    {
        auto candidate = std::numeric_limits<typename std::iterator_traits<ForwardIterator>::value_type>::lowest();
        for (; first != last; ++first)
        {
            if (candidate < *first)
            {
                candidate = *first;
            }
        }
        return candidate;
    }

    template <typename T>
    std::vector<T> make_input(std::size_t n, T constant)
    {
        std::vector<T> v(n);
        std::fill(v.begin(), v.end(), constant);
        return v;
    }

    template <typename T>
    void BM_std_max_element(benchmark::State& state)
    {
        const auto constant = T { 5 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        src[state.range(0)-1] = T { 10 }; // Ensure there is a maximum value at the end

        for (auto _ : state)
        {
            auto result = std::max_element(src.begin(), src.end());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
    
    template <typename T>
    void BM_xsimd_max_element(benchmark::State& state)
    {
        const auto constant = T { 5 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        src[state.range(0)-1] = T { 10 }; // Ensure there is a maximum value at the end
        for (auto _ : state)
        {
            auto result = xsimd::max_element(src.begin(), src.end());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    template <typename T>
    void BM_std_min_element(benchmark::State& state)
    {
        const auto constant = T { 5 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        src[state.range(0)-1] = T { 10 }; // Ensure there is a maximum value at the end

        for (auto _ : state)
        {
            auto result = std::min_element(src.begin(), src.end());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
    
    template <typename T>
    void BM_xsimd_min_element(benchmark::State& state)
    {
        const auto constant = T { 5 };
        auto src = make_input<T>(static_cast<std::size_t>(state.range(0)), constant);
        src[state.range(0)-1] = T { 10 }; // Ensure there is a maximum value at the end
        for (auto _ : state)
        {
            auto result = xsimd::min_element(src.begin(), src.end());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    static constexpr long long kSmall = 1LL << 12;
    static constexpr long long kMedium = 1LL << 18;
    static constexpr long long kLarge = 1LL << 22;

    // max_element
    BENCHMARK_TEMPLATE(BM_std_max_element, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_max_element, float)
    BENCH_ARGS;

    // min_element
    BENCHMARK_TEMPLATE(BM_std_min_element, float)
    BENCH_ARGS;
    BENCHMARK_TEMPLATE(BM_xsimd_min_element, float)
    BENCH_ARGS;

}