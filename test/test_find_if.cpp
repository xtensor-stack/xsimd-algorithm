/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd_algorithm/stl/find_if.hpp"


#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

#include <cstddef>
#include <numeric>
#include <vector>

#if XSIMD_WITH_NEON && !XSIMD_WITH_NEON64
#define FIND_IF_TYPES int, float
#else
#define FIND_IF_TYPES int, long, float, double
#endif


template <typename Type>
struct find_if_test
{
    using vector = std::vector<Type>;
    static constexpr std::size_t simd_size = xsimd::batch<Type>::size;

    static vector fill_data(size_t size, size_t position, Type constant)
    {
        vector result(size);
        std::iota(begin(result), end(result), Type(1));
        if (position < size)
        {
            result[position] = constant;
        }
        return result;
    }

    void test_find_if_aligned( size_t size, Type constant= {-42}) const
    {
        const auto position = size / 2;
        const vector data = fill_data(size, position, constant);
        
        const auto result = xsimd::find_if(data.begin(), data.end(), [constant](const auto& x) { return x == constant; });
        const auto expected = std::find_if(data.begin(), data.end(), [constant](const auto& x) { return x == constant; });
        CHECK(result == expected);
    }

    void test_find_if_misaligned(size_t size, Type constant= {-42}) const
    {
        const auto position = size / 2;
        const size_t missalignment = 1;
        const vector data = fill_data(size + missalignment * 2, position, constant);
        auto first = data.begin() + missalignment;
        auto last = first + size;

        const auto result = xsimd::find_if(first, last, [constant](const auto& x) { return x == constant; });
        const auto expected = std::find_if(first, last, [constant](const auto& x) { return x == constant; });

        CHECK(result == expected);
    }
};

TEST_CASE_TEMPLATE("find_if test", T, FIND_IF_TYPES)
{
    using Test = find_if_test<T>;

    const std::array test_sizes {
        size_t { 0 },
        size_t { 1 },
        Test::simd_size - 1,
        Test::simd_size,
        Test ::simd_size + 1,
    };

    Test test;

    size_t sub_case_id = 0;
    for (const auto size : test_sizes)
    {
        const std::string aligned_subcase_name = "Aligned" + std::to_string(sub_case_id++) + ", size: " + std::to_string(size);
        SUBCASE(aligned_subcase_name.c_str())
        {
            test.test_find_if_aligned(size);
        }
        const std::string misaligned_subcase_name = "Misaligned" + std::to_string(sub_case_id++) + ", size: " + std::to_string(size);
        SUBCASE(misaligned_subcase_name.c_str())
        {
            test.test_find_if_misaligned(size);
        }
    }
}
#endif