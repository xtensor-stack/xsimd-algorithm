/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_BENCHMARK_BENCH_ARENA_HPP
#define XSIMD_ALGORITHM_BENCHMARK_BENCH_ARENA_HPP

#include <cstddef>
#include <cstdint>
#include <new>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace xsimd::bench
{
    namespace detail
    {
#if defined(_WIN32)
        struct windows_virtual_memory
        {
            static auto map(std::size_t bytes) -> void*
            {
                void* raw = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                if (raw == nullptr)
                {
                    throw std::bad_alloc();
                }
                return raw;
            }

            /// Large pages require the SeLockMemoryPrivilege, not worth it for benchmarks.
            static void advise_huge_pages(void*, std::size_t) { }
        };

        using virtual_memory = windows_virtual_memory;
#else
        struct posix_virtual_memory
        {
            static auto map(std::size_t bytes) -> void*
            {
                void* raw = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (raw == MAP_FAILED)
                {
                    throw std::bad_alloc();
                }
                return raw;
            }

            static void advise_huge_pages([[maybe_unused]] void* ptr, [[maybe_unused]] std::size_t bytes)
            {
#ifdef MADV_HUGEPAGE
                ::madvise(ptr, bytes, MADV_HUGEPAGE);
#endif
            }
        };

        using virtual_memory = posix_virtual_memory;
#endif
    }

    /// A single memory region from which all benchmark buffers are carved.
    ///
    /// Throughput of memory-bound benchmarks depends on the physical pages backing the
    /// buffers (cache set and slice mapping, huge pages). Reusing the same region at the
    /// same offsets for every benchmark gives all variants the same physical layout.
    class arena
    {
    public:

        static constexpr std::size_t slot_alignment = std::size_t { 2 } << 20;
        static constexpr std::size_t capacity = std::size_t { 1 } << 30;

        static auto instance() -> arena&
        {
            static arena a;
            return a;
        }

        /// Release all buffers, the next allocation starts again at the beginning.
        void reset() { m_cursor = 0; }

        /// Return a pointer starting at a fresh slot, shifted by offset bytes.
        auto allocate(std::size_t bytes, std::size_t offset) -> std::byte*
        {
            auto const slot_bytes = round_up(bytes + offset);
            if (m_cursor + slot_bytes > capacity)
            {
                throw std::bad_alloc();
            }
            auto* ptr = m_base + m_cursor + offset;
            m_cursor += slot_bytes;
            return ptr;
        }

        arena(arena const&) = delete;
        arena& operator=(arena const&) = delete;

    private:

        static constexpr auto round_up(std::size_t bytes) -> std::size_t
        {
            return (bytes + slot_alignment - 1) & ~(slot_alignment - 1);
        }

        // Virtual memory only, pages are committed on first touch and never unmapped.
        arena()
        {
            auto const reserved = capacity + slot_alignment;
            void* raw = detail::virtual_memory::map(reserved);
            auto const address = round_up(reinterpret_cast<std::uintptr_t>(raw));
            m_base = reinterpret_cast<std::byte*>(address);
            detail::virtual_memory::advise_huge_pages(m_base, capacity);
        }

        std::byte* m_base = nullptr;
        std::size_t m_cursor = 0;
    };

    /// Allocator carving from the arena, @p Offset elements past a slot boundary.
    ///
    /// A zero offset is aligned for any architecture, a non-zero one gives unaligned buffers.
    /// Deallocation is a no-op, memory is reclaimed with arena::reset.
    template <typename T, std::size_t Offset = 0>
    struct arena_allocator
    {
        using value_type = T;

        template <typename U>
        struct rebind
        {
            using other = arena_allocator<U, Offset>;
        };

        arena_allocator() = default;

        template <typename U>
        arena_allocator(arena_allocator<U, Offset> const&)
        {
        }

        T* allocate(std::size_t n)
        {
            return reinterpret_cast<T*>(arena::instance().allocate(n * sizeof(T), Offset * sizeof(T)));
        }

        void deallocate(T*, std::size_t) { }

        friend bool operator==(arena_allocator const&, arena_allocator const&) { return true; }
    };
}

#endif
