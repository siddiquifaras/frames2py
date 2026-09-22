/**
 * @file module.cpp
 * @brief pybind11 module exposing C++ accumulation kernels to Python.
 *
 * The module `_frames2py_native` provides functions for each kernel
 * operation.  The GIL is released during the hot loops (accumulate)
 * so that consumer threads (viewer, telemetry) can run concurrently.
 *
 * Array arguments use pybind11's buffer protocol to obtain raw pointers
 * with zero-copy access to NumPy arrays.  Shape validation is performed
 * at the Python/C++ boundary, not inside the inner loops.
 *
 * Build:
 *   cd native && cmake -B build -DPYTHON_EXECUTABLE=$(which python)
 *   cmake --build build
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "frames2py/types.h"

namespace py = pybind11;

/* ------------------------------------------------------------------ */
/* Forward declarations for kernel implementations                      */
/* ------------------------------------------------------------------ */

namespace frames2py {

// event_count.cpp
class EventCountKernel;

// polarity.cpp
class PolarityKernel;

// time_surface.cpp
namespace time_surface {
void accumulate(const Event*, size_t, double*, uint32_t, uint32_t);
void snapshot(const double*, double*, uint32_t, uint32_t);
void reset(double*, uint32_t, uint32_t);
}  // namespace time_surface

}  // namespace frames2py

/* ------------------------------------------------------------------ */
/* Helper: extract Event pointer from a structured NumPy array          */
/* ------------------------------------------------------------------ */

static const frames2py::Event* get_events(
    py::array& arr,
    size_t& n_events
) {
    auto info = arr.request();
    if (info.ndim != 1) {
        throw std::runtime_error(
            "events array must be 1-D, got " + std::to_string(info.ndim) + "-D"
        );
    }
    if (info.itemsize != sizeof(frames2py::Event)) {
        throw std::runtime_error(
            "events itemsize mismatch: expected " +
            std::to_string(sizeof(frames2py::Event)) +
            ", got " + std::to_string(info.itemsize)
        );
    }
    n_events = static_cast<size_t>(info.shape[0]);
    return static_cast<const frames2py::Event*>(info.ptr);
}

/* ------------------------------------------------------------------ */
/* Event count kernel bindings                                          */
/* ------------------------------------------------------------------ */

static void py_accumulate_event_count(
    py::array events_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast>& state_arr,
    uint32_t width,
    uint32_t height
) {
    size_t n;
    const auto* events = get_events(events_arr, n);

    auto state_buf = state_arr.mutable_unchecked<2>();
    if (static_cast<uint32_t>(state_buf.shape(0)) != height ||
        static_cast<uint32_t>(state_buf.shape(1)) != width) {
        throw std::runtime_error("state shape mismatch");
    }
    float* state = state_arr.mutable_data();

    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t x = events[i].x;
            const uint32_t y = events[i].y;
            if (x < width && y < height) {
                state[y * width + x] += 1.0f;
            }
        }
    }
}

static void py_snapshot_event_count(
    py::array_t<float, py::array::c_style>& state_arr,
    py::array_t<float, py::array::c_style>& out_arr,
    uint32_t width,
    uint32_t height
) {
    const size_t total = static_cast<size_t>(width) * height;
    float* state = state_arr.mutable_data();
    float* out = out_arr.mutable_data();

    {
        py::gil_scoped_release release;
        std::memcpy(out, state, total * sizeof(float));
        std::memset(state, 0, total * sizeof(float));
    }
}

/* ------------------------------------------------------------------ */
/* Polarity kernel bindings                                             */
/* ------------------------------------------------------------------ */

static void py_accumulate_polarity(
    py::array events_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast>& state_arr,
    uint32_t width,
    uint32_t height
) {
    size_t n;
    const auto* events = get_events(events_arr, n);
    float* state = state_arr.mutable_data();

    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t x = events[i].x;
            const uint32_t y = events[i].y;
            const uint32_t p = events[i].p;
            if (x < width && y < height && p < 2) {
                state[(y * width + x) * 2 + p] += 1.0f;
            }
        }
    }
}

static void py_snapshot_polarity(
    py::array_t<float, py::array::c_style>& state_arr,
    py::array_t<float, py::array::c_style>& out_arr,
    uint32_t width,
    uint32_t height
) {
    const size_t total = static_cast<size_t>(width) * height * 2;
    float* state = state_arr.mutable_data();
    float* out = out_arr.mutable_data();

    {
        py::gil_scoped_release release;
        std::memcpy(out, state, total * sizeof(float));
        std::memset(state, 0, total * sizeof(float));
    }
}

/* ------------------------------------------------------------------ */
/* Time surface kernel bindings                                         */
/* ------------------------------------------------------------------ */

static void py_accumulate_time_surface(
    py::array events_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast>& state_arr,
    uint32_t width,
    uint32_t height
) {
    size_t n;
    const auto* events = get_events(events_arr, n);
    double* state = state_arr.mutable_data();

    {
        py::gil_scoped_release release;
        frames2py::time_surface::accumulate(events, n, state, width, height);
    }
}

static void py_snapshot_time_surface(
    py::array_t<double, py::array::c_style>& state_arr,
    py::array_t<double, py::array::c_style>& out_arr,
    uint32_t width,
    uint32_t height
) {
    const double* state = state_arr.data();
    double* out = out_arr.mutable_data();

    {
        py::gil_scoped_release release;
        frames2py::time_surface::snapshot(state, out, width, height);
    }
}

/* ------------------------------------------------------------------ */
/* Generic reset                                                        */
/* ------------------------------------------------------------------ */

static void py_reset_float(
    py::array_t<float, py::array::c_style>& arr,
    uint32_t width,
    uint32_t height
) {
    float* ptr = arr.mutable_data();
    const size_t total = static_cast<size_t>(arr.size());
    py::gil_scoped_release release;
    std::memset(ptr, 0, total * sizeof(float));
}

static void py_reset_double(
    py::array_t<double, py::array::c_style>& arr,
    uint32_t width,
    uint32_t height
) {
    double* ptr = arr.mutable_data();
    const size_t total = static_cast<size_t>(arr.size());
    py::gil_scoped_release release;
    std::memset(ptr, 0, total * sizeof(double));
}

/* ------------------------------------------------------------------ */
/* Module definition                                                    */
/* ------------------------------------------------------------------ */

PYBIND11_MODULE(_frames2py_native, m) {
    m.doc() = "frames2py C++ accumulation kernels with GIL release";

    // Event count
    m.def("accumulate_event_count", &py_accumulate_event_count,
          py::arg("events"), py::arg("state"),
          py::arg("width"), py::arg("height"),
          "Accumulate event counts into state[y, x]. GIL released.");

    m.def("snapshot_event_count", &py_snapshot_event_count,
          py::arg("state"), py::arg("out"),
          py::arg("width"), py::arg("height"),
          "Copy state to out and zero state. GIL released.");

    // Polarity
    m.def("accumulate_polarity", &py_accumulate_polarity,
          py::arg("events"), py::arg("state"),
          py::arg("width"), py::arg("height"),
          "Accumulate polarity events into 2-channel state. GIL released.");

    m.def("snapshot_polarity", &py_snapshot_polarity,
          py::arg("state"), py::arg("out"),
          py::arg("width"), py::arg("height"),
          "Copy 2-channel state to out and zero. GIL released.");

    // Time surface
    m.def("accumulate_time_surface", &py_accumulate_time_surface,
          py::arg("events"), py::arg("state"),
          py::arg("width"), py::arg("height"),
          "Update time surface with max timestamps. GIL released.");

    m.def("snapshot_time_surface", &py_snapshot_time_surface,
          py::arg("state"), py::arg("out"),
          py::arg("width"), py::arg("height"),
          "Copy time surface to out (no reset). GIL released.");

    // Reset
    m.def("reset_float", &py_reset_float,
          py::arg("arr"), py::arg("width"), py::arg("height"),
          "Zero a float32 state array. GIL released.");

    m.def("reset_double", &py_reset_double,
          py::arg("arr"), py::arg("width"), py::arg("height"),
          "Zero a float64 state array. GIL released.");
}
