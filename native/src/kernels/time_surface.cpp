/**
 * @file time_surface.cpp
 * @brief Time-surface accumulation kernel.
 *
 * Stores the **most recent** event timestamp per pixel in a
 * (height, width) double-precision buffer.  For each event, sets
 * state[y * width + x] = max(current, event.t).
 *
 * Unlike event_count and polarity, the time surface is **persistent**:
 * snapshot copies without resetting.
 *
 * Uses double (float64) state because timestamps are uint64 microseconds
 * and float32 loses precision above ~16M.
 *
 * Complexity: O(n) accumulate, O(w*h) snapshot.
 */

#include "frames2py/types.h"
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace frames2py {
namespace time_surface {

/**
 * Accumulate events into a double-precision time surface.
 */
void accumulate(
    const Event* events,
    size_t n,
    double* state,
    uint32_t width,
    uint32_t height
) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t x = events[i].x;
        const uint32_t y = events[i].y;
        if (x < width && y < height) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            const double t = static_cast<double>(events[i].t);
            if (t > state[idx]) {
                state[idx] = t;
            }
        }
    }
}

/**
 * Snapshot: copy state to output (no reset).
 */
void snapshot(
    const double* state,
    double* out,
    uint32_t width,
    uint32_t height
) {
    const size_t total = static_cast<size_t>(width) * height;
    std::memcpy(out, state, total * sizeof(double));
}

/**
 * Reset: zero the state buffer.
 */
void reset(
    double* state,
    uint32_t width,
    uint32_t height
) {
    const size_t total = static_cast<size_t>(width) * height;
    std::memset(state, 0, total * sizeof(double));
}

}  // namespace time_surface
}  // namespace frames2py
