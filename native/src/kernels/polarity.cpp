/**
 * @file polarity.cpp
 * @brief Two-channel polarity accumulation kernel.
 *
 * Maintains a (height, width, 2) state buffer:
 *   - channel 0: ON events (polarity == 1)
 *   - channel 1: OFF events (polarity == 0)
 *
 * State layout (row-major): state[(y * width + x) * 2 + channel].
 * Snapshot copies state to output and zeros state.
 *
 * Complexity: same as event_count -- O(n) accumulate, O(w*h) snapshot.
 */

#include "frames2py/kernel.h"
#include <cstring>

namespace frames2py {

class PolarityKernel : public Kernel {
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
            const uint32_t p = events[i].p;
            if (x < width && y < height && p < 2) {
                state[(y * width + x) * 2 + p] += 1.0f;
            }
        }
    }

    void snapshot(
        float* state,
        float* out,
        uint32_t width,
        uint32_t height
    ) override {
        const size_t total = static_cast<size_t>(width) * height * 2;
        std::memcpy(out, state, total * sizeof(float));
        std::memset(state, 0, total * sizeof(float));
    }

    void reset(
        float* state,
        uint32_t width,
        uint32_t height
    ) override {
        const size_t total = static_cast<size_t>(width) * height * 2;
        std::memset(state, 0, total * sizeof(float));
    }
};

}  // namespace frames2py
