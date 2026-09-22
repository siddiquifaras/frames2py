/**
 * @file kernel.h
 * @brief Abstract kernel interface for C++ accumulation backends.
 *
 * This header is the C++ expression of docs/spec/kernel_contract.md.
 * All kernel implementations derive from this base class and override
 * the three virtual methods.
 *
 * Design constraints:
 *   - accumulate() must not allocate (no new, no malloc, no std::vector resize).
 *   - accumulate() must not access any shared state beyond its parameters.
 *   - snapshot() copies state into out.  May optionally reset state.
 *   - All methods are safe to call without the Python GIL held.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "frames2py/types.h"

namespace frames2py {

class Kernel {
public:
    virtual ~Kernel() = default;

    /**
     * Accumulate events into the state buffer.
     *
     * @param events  Pointer to contiguous array of Event structs.
     * @param n       Number of events.
     * @param state   Pre-allocated accumulator (row-major, height*width for
     *                single-channel, or height*width*channels for multi).
     * @param width   Sensor width in pixels.
     * @param height  Sensor height in pixels.
     *
     * Complexity: O(n) where n = number of events.
     */
    virtual void accumulate(
        const Event* events,
        size_t n,
        float* state,
        uint32_t width,
        uint32_t height
    ) = 0;

    /**
     * Copy current state into the output buffer.
     *
     * @param state  Source accumulator.
     * @param out    Destination buffer (same layout as state).
     * @param width  Sensor width.
     * @param height Sensor height.
     *
     * Whether state is reset after copy is kernel-dependent.
     */
    virtual void snapshot(
        float* state,
        float* out,
        uint32_t width,
        uint32_t height
    ) = 0;

    /**
     * Zero the accumulator state.
     *
     * @param state  Accumulator to reset.
     * @param width  Sensor width.
     * @param height Sensor height.
     */
    virtual void reset(
        float* state,
        uint32_t width,
        uint32_t height
    ) = 0;
};

}  // namespace frames2py
