/**
 * @file types.h
 * @brief Packed event struct matching Python EVENT_DTYPE exactly.
 *
 * The Event struct is 13 bytes packed:
 *   - t: uint64_t  (timestamp in microseconds)
 *   - x: uint16_t  (pixel x coordinate)
 *   - y: uint16_t  (pixel y coordinate)
 *   - p: uint8_t   (polarity, 0 or 1)
 *
 * This must match the NumPy structured dtype layout byte-for-byte so
 * that pybind11 can pass event arrays via the buffer protocol without
 * any conversion or copy.
 *
 * Memory layout (little-endian):
 *   offset 0:  t  (8 bytes)
 *   offset 8:  x  (2 bytes)
 *   offset 10: y  (2 bytes)
 *   offset 12: p  (1 byte)
 *   total:        13 bytes
 */

#pragma once

#include <cstdint>

namespace frames2py {

#pragma pack(push, 1)
struct Event {
    uint64_t t;   ///< Timestamp in microseconds.
    uint16_t x;   ///< Pixel x coordinate.
    uint16_t y;   ///< Pixel y coordinate.
    uint8_t  p;   ///< Polarity (0 = OFF, 1 = ON).
};
#pragma pack(pop)

static_assert(sizeof(Event) == 13, "Event struct must be exactly 13 bytes to match EVENT_DTYPE");
static_assert(alignof(Event) == 1, "Event struct must have alignment 1 (packed)");

}  // namespace frames2py
