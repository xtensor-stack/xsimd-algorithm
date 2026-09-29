/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <array>
#include <cstddef>
#include <cstdint>

#include <doctest/doctest.h>
#include <xsimd_test_utils/utils.hpp>

#include "xsimd_algorithm/map.hpp"

/// Map unary test where one input batch pairs with two output batches.
TEST_CASE("map_unary int32 to int64")
{
    using input_type = std::int32_t;
    using output_type = std::int64_t;

    // Not a multiple of the batch size, to exercise the tail.
    static constexpr std::size_t size = 94;

    const auto input = xsimd::test::make_arange<input_type>(size);
    auto output = xsimd::test::aligned_vector<output_type>(size);

    const auto func = [](auto const& x)
    { return xsimd::widen(x + input_type { 1 }); };

    xsimd::map_unary(input, output, func);

    for (std::size_t i = 0; i < size; ++i)
    {
        CAPTURE(i);
        CHECK(output[i] == static_cast<output_type>(input[i] + 1));
    }
}

/// Map unary test where two input batches pair with one output batch.
TEST_CASE("map_unary int64 to int32")
{
    using input_type = std::int64_t;
    using output_type = std::int32_t;
    using input_batch = xsimd::batch<input_type>;
    using output_batch = xsimd::batch<output_type>;

    // Not a multiple of the batch size, to exercise the tail.
    static constexpr std::size_t size = 94;

    const auto input = xsimd::test::make_arange<input_type>(size);
    auto output = xsimd::test::aligned_vector<output_type>(size);

    // xsimd has no narrowing counterpart to widen, so truncate by keeping the low
    // half of each lane, those of the first batch followed by those of the second.
    struct low_halves
    {
        static constexpr unsigned get(unsigned i, unsigned n)
        {
            return (i < n / 2) ? 2 * i : n + 2 * (i - n / 2);
        }
    };

    const auto func = [](std::array<input_batch, 2> const& x) -> output_batch
    {
        return xsimd::shuffle(
            xsimd::bitwise_cast<output_type>(x[0] + input_type { 1 }),
            xsimd::bitwise_cast<output_type>(x[1] + input_type { 1 }),
            xsimd::make_batch_constant<std::uint32_t, low_halves>());
    };

    xsimd::map_unary(xsimd::test::as_span(input), xsimd::test::as_span(output), func);

    for (std::size_t i = 0; i < size; ++i)
    {
        CAPTURE(i);
        CHECK(output[i] == static_cast<output_type>(input[i] + 1));
    }
}
