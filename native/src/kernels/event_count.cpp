/**
 * @file event_count.cpp
 * @brief Event-count accumulation kernel.
 *
 * Counts events per pixel.  For each event, increments
 * state[y * width + x] by 1.0.  Snapshot copies state to output and
 * zeros state.
 *
 * Complexity:
 *   accumulate: O(n) where n = number of events.
 *   snapshot:   O(width * height) -- two memcpy-class operations.
 *   reset:      O(width * height).
 *
 * No dynamic allocation.  No shared state.  No locks.
 */

#include "frames2py/kernel.h"
#include <cstring>
#include <algorithm>

namespace frames2py {

class EventCountKernel : public Kernel {
public:
    void accumulate(
        const Event* events,
        size_t n,
        float* state,
        uint32_t width,
        uint32_t height
    ) override {
        for (size_t i = 0; i < n; ++i) {
            const uint32_t x = events[i].x;
            const uint32_t y = events[i].y;
            // Bounds check: silently skip out-of-range coordinates.
            if (x < width && y < height) {
                state[y * width + x] += 1.0f;
            }
        }
    }

    void snapshot(
        float* state,
        float* out,
        uint32_t width,
        uint32_t height
    ) override {
        const size_t total = static_cast<size_t>(width) * height;
        std::memcpy(out, state, total * sizeof(float));
        std::memset(state, 0, total * sizeof(float));
    }

    void reset(
        float* state,
        uint32_t width,
        uint32_t height
    ) override {
        const size_t total = static_cast<size_t>(width) * height;
        std::memset(state, 0, total * sizeof(float));
    }
};

}  // namespace frames2py
