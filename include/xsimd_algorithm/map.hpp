/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_MAP_HPP
#define XSIMD_ALGORITHM_MAP_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include <xsimd/xsimd.hpp>

#include "./macros.hpp"

namespace xsimd
{
    /// Return the pointer before the input with the given alignment or itself if aligned.
    template <typename T>
    XSIMD_INLINE auto prev_aligned(T* ptr, std::size_t alignment) -> T*
    {
        assert(std::has_single_bit(alignment));
        auto const address = reinterpret_cast<std::size_t>(ptr);
        return reinterpret_cast<T*>(address & ~(alignment - 1));
    }

    /// Return the pointer after the input with the given alignment or itself if aligned.
    template <typename T>
    XSIMD_INLINE auto next_aligned(T* ptr, std::size_t alignment) -> T*
    {
        assert(std::has_single_bit(alignment));
        auto const address = reinterpret_cast<std::size_t>(ptr);
        return reinterpret_cast<T*>((address + alignment - 1) & ~(alignment - 1));
    }

    template <typename T>
    XSIMD_INLINE auto bytes_to_next_aligned(T* ptr, std::size_t alignment) -> std::size_t
    {
        assert(std::has_single_bit(alignment));
        auto const address = reinterpret_cast<std::uintptr_t>(ptr);
        return (alignment - (address & (alignment - 1))) & (alignment - 1);
    }

    /// Check if two spans are aliasing each others (overlapping).
    template <typename T, std::size_t et, typename U, std::size_t eu>
    XSIMD_INLINE auto are_aliased(std::span<T, et> lhs, std::span<U, eu> rhs) -> bool
    {
        // Comparing pointers from unrelated objects is unspecified, integers are not.
        auto const lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data());
        auto const rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data());
        return (lhs_begin < rhs_begin + rhs.size_bytes()) && (rhs_begin < lhs_begin + lhs.size_bytes());
    }

    /// Copy fewer than 2 * k elements without a call to memcpy.
    ///
    /// For count in [k, 2k), two overlapping copies of k elements cover the range; since k is a
    /// compile-time constant, each memcpy compiles down to a few fixed-size loads and stores.
    template <std::size_t k, typename T>
    XSIMD_INLINE void copy_small(T* XSIMD_RESTRICT dst, T const* XSIMD_RESTRICT src, std::size_t count)
    {
        assert(std::has_single_bit(k));
        assert(count < 2 * k);
        if (count >= k)
        {
            std::memcpy(dst, src, k * sizeof(T));
            std::memcpy(dst + count - k, src + count - k, k * sizeof(T));
        }
        else if (k > 1)
        {
            copy_small<k / 2>(dst, src, count);
        }
    }

    /// Load batch wrapper with an alignment as template parameter.
    template <typename T, typename A, bool aligned>
    XSIMD_INLINE xsimd::batch<T, A> load_batch(T const* ptr)
    {
        if constexpr (aligned)
        {
            return xsimd::batch<T, A>::load_aligned(ptr);
        }
        else
        {
            return xsimd::batch<T, A>::load_unaligned(ptr);
        }
    }

    /// Store batch wrapper with an alignment as template parameter.
    template <typename T, typename A, bool aligned>
    XSIMD_INLINE void store_batch(xsimd::batch<T, A> x, T* ptr)
    {
        if constexpr (aligned)
        {
            x.store_aligned(ptr);
        }
        else
        {
            x.store_unaligned(ptr);
        }
    }

    struct alignment_options
    {
        bool start_aligned = false;
        bool end_aligned = false;
    };

    struct map_options
    {
        std::size_t unroll_factor = 4;
        bool pure = false;
    };

    namespace internal
    {
        template <typename T>
        inline constexpr bool is_array = false;

        template <typename T, std::size_t N>
        inline constexpr bool is_array<std::array<T, N>> = true;

        /// If an array contains only one element, return it.
        template <typename B, std::size_t N>
        XSIMD_INLINE auto const& unwrap_array(std::array<B, N> const& x)
        {
            if constexpr (N == 1)
            {
                return x[0];
            }
            else
            {
                return x;
            }
        }

        /// Wrap user function to handle ``xsimd::batch`` as 1D array.
        ///
        /// Transform 1D input array as batch from algorithm functions to batch for to
        /// the user function, and user batch result as 1D arrays for the algorithm
        /// functions.
        template <typename Func>
        XSIMD_INLINE auto wrap_params_as_1d_arrays(Func&& func)
        {
            return [func = std::forward<Func>(func)](auto const&... x)
            {
                auto res = func(internal::unwrap_array(x)...);
                if constexpr (internal::is_array<decltype(res)>)
                {
                    return res;
                }
                else
                {
                    return std::array { res };
                }
            };
        }

        template <map_options opts, alignment_options align, typename A, typename Out, typename... In>
        struct map_helper
        {
            static constexpr std::size_t n_input = sizeof...(In);
            static constexpr auto input_elem_size = std::array { sizeof(In)... };

            static constexpr std::size_t min_elem_size = std::min({ sizeof(Out), sizeof(In)... });
            static constexpr std::size_t max_elem_size = std::max({ sizeof(Out), sizeof(In)... });

            /// Inputs and output may not have the same alignment so it may be impossible
            /// to get all aligned. We align preferably the output (more expensive unaligned
            /// stores) or otherwise one of the input.
            static constexpr bool align_output = sizeof(Out) == max_elem_size;

            static constexpr std::array<bool, n_input> get_align_inputs()
            {
                std::array<bool, n_input> out {};
                bool found = false;
                for (std::size_t k = 0; k < out.size(); ++k)
                {
                    if (found || align_output)
                    {
                        out[k] = false;
                    }
                    else
                    {
                        found = input_elem_size[k] == max_elem_size;
                        out[k] = found;
                    }
                }
                return out;
            }

            static constexpr std::array<bool, n_input> align_inputs = get_align_inputs();

            static constexpr bool output_is_aligned = align_output || align.start_aligned;

            static constexpr std::array<bool, n_input> get_input_is_aligned()
            {
                std::array<bool, n_input> out = align_inputs;
                for (bool& b : out)
                {
                    b = b || align.start_aligned;
                }
                return out;
            }

            static constexpr std::array<bool, n_input> input_is_aligned = get_input_is_aligned();

            /// Number of batches of T spanning as many elements as one batch of the widest element.
            ///
            /// Pairing that many batches on all side lets both sides advance by the same number of
            /// elements, so a mapping stays elementwise regardless of the respective lane counts.
            template <typename T>
            static constexpr std::size_t batch_arity()
            {
                return sizeof(T) / min_elem_size;
            }

            template <typename T>
            using batch_array = std::array<xsimd::batch<T, A>, batch_arity<T>()>;

            static constexpr std::size_t chunk_size = batch_arity<Out>() * xsimd::batch<Out, A>::size;

            /// Load an array of batches.
            template <bool aligned, typename T>
            XSIMD_INLINE static auto load_batches(T const* ptr) -> batch_array<T>
            {
                batch_array<T> x;
                for (std::size_t i = 0; i < x.size(); ++i)
                {
                    x[i] = load_batch<T, A, aligned>(ptr + i * xsimd::batch<T, A>::size);
                }
                return x;
            }

            /// Store an array of batches.
            template <bool aligned, typename T>
            XSIMD_INLINE static void store_batches(batch_array<T> const& x, T* ptr)
            {
                for (std::size_t i = 0; i < x.size(); ++i)
                {
                    store_batch<T, A, aligned>(x[i], ptr + i * xsimd::batch<T, A>::size);
                }
            }

            /// Map a single unaligned chunk.
            template <typename Func>
            XSIMD_INLINE static void map_chunk_unaligned(
                In const* XSIMD_RESTRICT... in,
                Out* XSIMD_RESTRICT out,
                Func&& func)
            {
                store_batches<false>(func(load_batches<false>(in)...), out);
            }

            /// Map multiple chunks with a compile-time unrolled loop.
            template <
                std::array<bool, n_input> load_aligned,
                bool store_aligned,
                typename Func,
                std::size_t... step>
            XSIMD_INLINE static void map_unrolled(
                In const* XSIMD_RESTRICT... in,
                Out* XSIMD_RESTRICT out,
                Func&& func,
                std::index_sequence<step...>)
            {
                static constexpr std::size_t factor = sizeof...(step);

                constexpr auto read = []<class T, std::size_t... a>(T const* ptr, std::index_sequence<a...>)
                {
                    std::array<batch_array<T>, factor> x;
                    auto load_one = [&](std::size_t s)
                    {
                        // Expand the parameter pack a for each alignment.
                        ((x[s] = load_batches<load_aligned[a]>(ptr + s * chunk_size)), ...);
                    };
                    // Expand the step parameter pack: repeat the unrolled operation.
                    (load_one(step), ...);
                    return x;
                };

                constexpr auto map = [](auto* out, auto&& f, auto const&... x)
                {
                    auto map_one = [&](std::size_t s)
                    {
                        // Expand the parameter pack a for each input.
                        store_batches<store_aligned>(f(x[s]...), out + s * chunk_size);
                    };
                    // Expand the step parameter pack: repeat the unrolled operation.
                    (map_one(step), ...);
                };

                map(out, std::forward<Func>(func), read(in, std::index_sequence_for<In...> {})...);
            }

            /// Map chunks in a loop with given unrolling factor.
            ///
            /// Return number of elements mapped.
            template <
                std::array<bool, n_input> load_aligned,
                bool store_aligned,
                std::size_t unroll_factor,
                typename Func>
            XSIMD_INLINE static auto map_loop(
                In const* XSIMD_RESTRICT... in,
                Out* XSIMD_RESTRICT out,
                std::size_t count,
                Func&& func) -> std::size_t
            {
                constexpr auto steps = std::make_index_sequence<unroll_factor>();
                constexpr std::size_t total_step_size = unroll_factor * chunk_size;

                std::size_t remaining = count;
                while (remaining >= total_step_size)
                {
                    map_unrolled<load_aligned, store_aligned>(in..., out, func, steps);
                    ((in += total_step_size), ...);
                    out += total_step_size;
                    remaining -= total_step_size;
                }
                return count - remaining;
            }

            /// Map fewer elements than a full chunk through a scratch buffer.
            template <typename Func>
            XSIMD_INLINE static void map_chunk_partial(
                In const* XSIMD_RESTRICT... begin,
                Out* XSIMD_RESTRICT out,
                std::size_t count,
                Func&& func)
            {
                assert(count <= chunk_size);
                if (count == 0) [[unlikely]]
                {
                    return;
                }

                constexpr auto read = []<typename T>(T const* in, std::size_t cnt)
                {
                    alignas(A::alignment()) std::array<T, chunk_size> in_buffer = {};
                    copy_small<chunk_size>(in_buffer.data(), in, cnt);
                    return load_batches<true>(in_buffer.data());
                };

                alignas(A::alignment()) std::array<Out, chunk_size> out_buffer;
                store_batches<true>(func(read(begin, count)...), out_buffer.data());
                copy_small<chunk_size>(out, out_buffer.data(), count);
            }

            /// Given some pointers, return the number of element to process until desired alignment.
            ///
            /// The desired alignment is given via the compile-time parameters.
            /// Only one can be true.
            template <std::array<bool, n_input> align_in, bool align_out>
            XSIMD_INLINE static auto elems_to_alignment(In const*... in, Out* out) -> std::size_t
            {
                if constexpr (align_out)
                {
                    return bytes_to_next_aligned(out, A::alignment()) / sizeof(Out);
                }
                else
                {
                    constexpr auto iter = std::find(align_in.begin(), align_in.end(), true);
                    static_assert(iter < align_in.end());
                    constexpr auto idx = iter - align_in.begin();
                    auto to_align = std::array { in... }[idx];
                    return bytes_to_next_aligned(to_align, A::alignment()) / input_elem_size[idx];
                }
            }

            template <typename Func>
            XSIMD_INLINE static void map_n(
                In const* XSIMD_RESTRICT... in,
                Out* XSIMD_RESTRICT out,
                std::size_t count,
                Func&& func)
            {
                const auto advance = [&](std::size_t n)
                {
                    ((in += n), ...);
                    out += n;
                    count -= n;
                };

                if (count == 0) [[unlikely]]
                {
                    return;
                }

                if constexpr (!align.start_aligned)
                {
                    // The span may be too short to reach the next alignment boundary.
                    const auto to_alignment = elems_to_alignment<align_inputs, align_output>(in..., out);
                    const auto head = std::min(to_alignment, count);

                    if (opts.pure && (head != 0) && (count >= chunk_size))
                    {
                        // Recompute the head as a full step, the body overwrites the excess.
                        map_chunk_unaligned(in..., out, func);
                    }
                    else
                    {
                        map_chunk_partial(in..., out, head, func);
                    }
                    advance(head);
                }

                // Unrolled loop processing multiple chunks at a time.
                auto processed = map_loop<input_is_aligned, output_is_aligned, opts.unroll_factor>(
                    in..., out, count, func);
                advance(processed);

                // Regular simd loop one chunk at a time.
                processed = map_loop<input_is_aligned, output_is_aligned, 1>(in..., out, count, func);
                advance(processed);

                // Unlikely to be skipped, meant for users that know they allocate
                // a multiple of the batch size, such as in a local buffer
                if constexpr (!align.end_aligned)
                {
                    if (opts.pure && (count != 0) && (count >= chunk_size)) [[likely]]
                    {
                        // Recompute overlapping data, this time starting from the end.
                        map_chunk_unaligned((in + count - chunk_size)..., out + count - chunk_size, func);
                    }
                    else
                    {
                        map_chunk_partial(in..., out, count, func);
                    }
                }
            }
        };
    }

    /// Apply func elementwise over in, writing as many elements to out.
    ///
    /// Func maps as many input as given to the function.
    /// If input as of different sizes, then the larger ones must be passed as an array of
    /// as many batches as the factor to the smallest element size, so that the function
    /// processes a fixed amount of elements.
    template <
        alignment_options align = alignment_options {},
        map_options opts = map_options {},
        typename Arch = xsimd::default_arch,
        typename Func,
        typename Out,
        typename... In>
    XSIMD_INLINE void map_n(Func&& func, Out&& out, In&&... in)
    {
        using H = internal::map_helper<
            opts,
            align,
            Arch,
            typename std::remove_cvref_t<Out>::value_type,
            typename std::remove_cvref_t<In>::value_type...>;

        assert((... && (in.size() == out.size())));
        assert((... && !are_aliased(std::span(in.data(), in.size()), std::span(out.data(), out.size()))));

        auto mapper = internal::wrap_params_as_1d_arrays(std::forward<Func>(func));
        return H::map_n(in.data()..., out.data(), out.size(), mapper);
    }

    template <
        alignment_options align = alignment_options {},
        map_options opts = map_options {},
        typename Arch = xsimd::default_arch,
        typename Func,
        typename Out,
        typename In>
    XSIMD_INLINE void map_unary(In&& in, Out&& out, Func&& func)
    {
        return map_n<align, opts, Arch>(func, out, in);
    }

    template <
        alignment_options align = alignment_options {},
        map_options opts = map_options {},
        typename Arch = xsimd::default_arch,
        typename Func,
        typename Out,
        typename Lhs,
        typename Rhs>
    XSIMD_INLINE void map_binary(Lhs&& lhs, Rhs&& rhs, Out&& out, Func&& func)
    {
        return map_n<align, opts, Arch>(func, out, lhs, rhs);
    }
}

#endif
