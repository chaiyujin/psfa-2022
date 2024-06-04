#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <limits>

using namespace pybind11::literals;
namespace py = pybind11;

struct int2 {
    int x = 0;
    int y = 0;

    int2 const & operator += (int2 const & _b) {
        x += _b.x;
        y += _b.y;
        return *this;
    }
    bool operator == (int2 const & _b) {
        return x == _b.x && y == _b.y;
    }
};

std::tuple<
    std::vector<int64_t>,
    std::vector<std::tuple<float, float, float>>,
    py::array_t<uint8_t, py::array::c_style>
> find_contour(
    py::array_t<float, py::array::c_style> rast_out,
    py::array_t<uint8_t, py::array::c_style> cntr_fidx_flags,
    py::array_t<uint8_t, py::array::c_style> face_cntr_fidx_flags
) {
    assert(rast_out.ndim() == 3);
    assert(cntr_fidx_flags.ndim() == 1);
    assert(face_cntr_fidx_flags.ndim() == 1);

    int H = (int)rast_out.shape(0);
    int W = (int)rast_out.shape(0);

    // * marking for contour and visited flags
    py::array_t<uint8_t, py::array::c_style> marking({H, W});
    memset(marking.mutable_data(), 0, marking.size() * sizeof(uint8_t));
    // * face contour
    std::vector<int64_t> faces_idx;
    std::vector<std::tuple<float, float, float>> bary_coords;

    auto _appendPossiblePosition = [&](const int2 & pos) -> void
    {
        int64_t idx_off = static_cast<int64_t>(rast_out.at(pos.y, pos.x, 3));
        // guard, which is not visible pixel at all
        if (idx_off == 0) return;

        int64_t fidx = idx_off - 1;
        if (face_cntr_fidx_flags.at(fidx)) {
            // is a valid face contour candidate
            float u = rast_out.at(pos.y, pos.x, 0);
            float v = rast_out.at(pos.y, pos.x, 1);
            faces_idx.push_back(fidx);
            bary_coords.push_back(std::make_tuple(u, v, 1.0f - u - v));
            marking.mutable_at(pos.y, pos.x) = (uint8_t)255;
        }
        else {
            // only on contour
            marking.mutable_at(pos.y, pos.x) = (uint8_t)50;
        }
    };

    // * find the first pixel on the contour
    int2 xy0 = {-1, -1};
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t idx = static_cast<size_t>(rast_out.at(y, x, 3));
            if (idx > 0 && cntr_fidx_flags.at(idx-1) > 0) {
                xy0 = {x, y};
                break;
            }
        }
    }

    // * find contour and record face contour
    if (xy0.x >= 0 && xy0.y >= 0) {
        // * (-1, -1)\ ( 0, -1) /( 1, -1)
        // *          \    |   /
        // * (-1,  0)<-( 0,  0)->( 1,  0)
        // *          /    |   \
        // * (-1,  1)/ ( 0,  1) \( 1,  1)

        static const std::vector<int2> s_goto_dirs = {
            { 1,  0}, { 1, -1}, { 0, -1}, {-1, -1},
            {-1,  0}, {-1,  1}, { 0,  1}, { 1,  1}
        };

        auto _nextDir = [&](const int2 & pos, size_t i_last_dir) -> size_t
        {
            size_t i_from = (i_last_dir + s_goto_dirs.size() / 2);
            for (size_t k = 1; k <= s_goto_dirs.size(); ++k) {
                size_t i = (i_from + k) % s_goto_dirs.size();
                int x = pos.x + s_goto_dirs[i].x;
                int y = pos.y + s_goto_dirs[i].y;
                if (0 <= x && x < W && 0 <= y && y < H) {
                    size_t idx = static_cast<size_t>(rast_out.at(y, x, 3));
                    if (idx > 0 && cntr_fidx_flags.at(idx-1) > 0) {
                        return i;
                    }
                }
            }
            // not found!
            return s_goto_dirs.size();
        };

        // first position
        int2 pos = xy0;
        _appendPossiblePosition(pos);
        const int n_pixels = H * W;

        // find all by linear search
        int n_visited_pixels = 0;
        size_t i_last_dir = 0;  // we will start to scan, the last dir is {1, 0}
        while (i_last_dir < s_goto_dirs.size()) {
            i_last_dir = _nextDir(pos, i_last_dir);
            // found
            if (i_last_dir < s_goto_dirs.size()) {
                pos += s_goto_dirs[i_last_dir];
                // break the loop in two conditions:
                // 1) find the first pos agian
                if (pos == xy0) { break; }
                // 2) more then number of pixels (it's error)
                if (++n_visited_pixels > n_pixels) {
                    faces_idx.clear();
                    bary_coords.clear();
                    printf(
                        "[contour]: loop over %d pixels, but only %d pixel in total. It must be wrong!\n",
                        n_visited_pixels, n_pixels
                    );
                    break;
                }

                // if (marking.valueAt(pos.x, pos.y) != 0) {
                //     break;
                // }

                // go on to find
                _appendPossiblePosition(pos);
            }
        }
    }

    return {faces_idx, bary_coords, marking};
}

PYBIND11_MODULE(contour_finder, m) {
    m.doc() = "pybind11 contour finder";
    m.def("find_contour", &find_contour, py::return_value_policy::move);
}
