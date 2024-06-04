#include <yuki/yuki.hpp>
#include <yuki/experimental.hpp>

namespace yuki {

template <typename Scalar>
struct BaryCoord
{
    vec<size_t, 3> vert_indices;
    vec<Scalar, 3> weights;

    template <typename T, int N>
    vec<T, N> operator()(const vec<T, N> * _list) const
    {
        return _list[vert_indices[0]] * (T)weights[0] +
               _list[vert_indices[1]] * (T)weights[1] +
               _list[vert_indices[2]] * (T)weights[2];
    }
};

}

#if defined(YUKI_MODULE_OPENGL)

namespace yuki::gl {

class ScreenSpaceFaceAreaHelper
{
private:
    bool                        inited_;
    std::shared_ptr<gl::Window> bg_window_;
    gl::TextureBuffer           result_buffer_;
    data::Image<float>          result_cntr_;
    gl::MeshComputeShading      mesh_compute_;
    std::vector<uint8_t>        mask_face_verts_;
    std::vector<uint8_t>        mask_cntr_tris_;

    mat<float, 4, 4>            mat_model_;
    mat<float, 4, 4>            mat_view_;
    mat<float, 4, 4>            mat_proj_;

    static int _GetIndexInt(float _i, float _n)
    {
        auto ti_f = _i * _n - 1.f;
        auto ti = static_cast<int>(std::round(ti_f));
        log::assertion(std::abs(ti-ti_f) < 1e-2, "not close to integer! {} {}\n", ti_f, ti);
        return ti;
    };

    static float _GetIndexFloat(int _i, int _n)
    {
        return (float)(_i + 1) / (float)_n;
    }

    void _fillDataOfBaryCoords()
    {
        // fill data of barycoords
        const size_t n_positions = mesh_compute_.mesh().numPositions();
        const size_t n_triangles = mesh_compute_.mesh().numTriangles();
        for (size_t ti = 0; ti < n_triangles; ++ti)
        {
            auto sum = (
                mask_face_verts_[mesh_compute_.mesh().indexAt(ti*3+0).x] +
                mask_face_verts_[mesh_compute_.mesh().indexAt(ti*3+1).x] +
                mask_face_verts_[mesh_compute_.mesh().indexAt(ti*3+2).x]
            );
            for (size_t k = 0; k < 3; ++k)
            {
                auto & mutable_data = mesh_compute_.triVertDataAt(ti*3+k);
                if (sum == 3)
                {
                    mutable_data = {
                        (float)(k==1), (float)(k==2),
                        _GetIndexFloat(ti, n_triangles), 1
                    };
                } else {
                    mutable_data = {0, 0, 0, 0};
                }
            }
        }
    }

    void _fillVertIndexData()
    {
        // fill data of vert index
        const size_t n_positions = mesh_compute_.mesh().numPositions();
        const size_t n_triangles = mesh_compute_.mesh().numTriangles();
        for (size_t ti = 0; ti < n_triangles; ++ti)
        {
            auto sum = (
                mask_face_verts_[mesh_compute_.mesh().indexAt(ti*3+0).x] +
                mask_face_verts_[mesh_compute_.mesh().indexAt(ti*3+1).x] +
                mask_face_verts_[mesh_compute_.mesh().indexAt(ti*3+2).x]
            );
            for (size_t k = 0; k < 3; ++k)
            {
                auto & mutable_data = mesh_compute_.triVertDataAt(ti*3+k);
                if (sum == 3)
                {
                    auto vi = mesh_compute_.mesh().indexAt(ti*3+k).x;
                    mutable_data = {
                        0, 0, _GetIndexFloat(vi, n_positions), 1
                    };
                } else {
                    mutable_data = {0, 0, 0, 0};
                }
            }
        }
    }

    void _maybeCreateWindow()
    {
        if (bg_window_ == nullptr)
        {
            gl::Initialize();
            gl::Window::Settings settings = {};
            settings.size = {512, 512};
            settings.swap_interval = 0;
            settings.flags -= gl::Window::kVisible;
            bg_window_ = std::make_shared<gl::Window>(settings);
            // log::info("[ScreenSpaceFaceAreaHelper]: create window for face helper. {} ({}x{})",
            //     bg_window_->title(), bg_window_->windowSize().x, bg_window_->windowSize().y
            // );

            bg_window_->onDraw(std::bind(&ScreenSpaceFaceAreaHelper::_meshCompute,
                this, std::placeholders::_1));
        }
    }

public:
    ScreenSpaceFaceAreaHelper()
        : inited_(false)
        , mat_model_(mat_zero<float, 4, 4>)
        , mat_view_ (mat_zero<float, 4, 4>)
        , mat_proj_ (mat_zero<float, 4, 4>) { _maybeCreateWindow(); }
    ~ScreenSpaceFaceAreaHelper() { this->resetOpenGLContextRelated(); }

    void resetOpenGLContextRelated()
    {
        this->mesh_compute_.resetOpenGLContextRelated();
        this->result_buffer_.reset();
        this->bg_window_.reset();
    }

    bool initialized() const { return inited_; }
    void reset() { inited_ = false; mask_face_verts_.clear(), mask_cntr_tris_.clear(); }
    const data::Mesh    & mesh()   const { return mesh_compute_.mesh();   }
          Camera<float> & camera()       { return mesh_compute_.camera(); }
    const Camera<float> & camera() const { return mesh_compute_.camera(); }
    void set_mesh  (const data::Mesh    & _mesh  ) { mesh_compute_.set_mesh(_mesh);     }
    void set_camera(const Camera<float> & _camera) { mesh_compute_.set_camera(_camera); }
    void set_mat_model(const mat<float, 4, 4> & _mat) { mat_model_ = _mat; }
    void set_mat_view (const mat<float, 4, 4> & _mat) { mat_view_  = _mat; }
    void set_mat_proj (const mat<float, 4, 4> & _mat) { mat_proj_  = _mat; }
    void initialize(const data::Mesh & _mesh,
                    std::string _path_of_face_area_vert_indices = "",
                    std::string _path_of_contour_area_tri_indices = "")
    {
        // guard
        if (inited_) return;

        inited_ = true;
        mesh_compute_.set_mesh(_mesh);
        mask_face_verts_.resize(_mesh.numPositions(), 1);
        mask_cntr_tris_ .resize(_mesh.numTriangles(), 1);

        auto _readMask = [](std::string _path, std::vector<uint8_t> & _indices)
        {
            if (_path.length() == 0) return;
            std::ifstream fin(_path);
            if (fin.is_open())
            {
                memset(_indices.data(), 0, _indices.size());
                while (!fin.eof()) {
                    int x; fin >> x; _indices[x] = 1;
                }
                fin.close();
            } else {
                log::error("Failed to load mask indices: {}", _path);
            }
        };
        _readMask(_path_of_face_area_vert_indices,   mask_face_verts_);
        _readMask(_path_of_contour_area_tri_indices, mask_cntr_tris_);
    }

    void _meshCompute(gl::Window *self)
    {
        // Timeit _("render frame");
        result_buffer_.begin({512, 512}, 1);
        {
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            _fillDataOfBaryCoords();
            mesh_compute_.updateData();
            mesh_compute_.draw(GL_TRIANGLES, self, result_buffer_.buffer_size(),
                            mat_model_, mat_view_, mat_proj_);
        }
        result_buffer_.end();
        result_cntr_.reshape(result_buffer_.buffer_size(), 4);
        glBindTexture(GL_TEXTURE_2D, result_buffer_.texture_id());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, (void*)result_cntr_.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    std::pair<std::vector<std::tuple<size_t, BaryCoord<double>, float2>>, data::Image<uint8_t>>
    barycoordsOnContour(std::optional<std::string> _debug_prefix = std::nullopt, bool _debug_must = false)
    {
        bg_window_->step();

        auto & result = result_cntr_;
        std::vector<std::tuple<size_t, BaryCoord<double>, float2>> barycoords_on_contour;
        data::Image<uint8_t> marking(result.resolution(), 1); marking.memset(0);  // cache the find pos
        float n_tri = (float)mesh_compute_.mesh().numTriangles();

        auto _appendPossiblePosition = [&](const int2 & pos) -> void
        {
            const auto & data = result.pixelAt<3>(pos.x, pos.y);
            auto ti = _GetIndexInt(data.z, n_tri);
            if (ti >= 0 && mask_cntr_tris_[ti])
            {
                barycoords_on_contour.push_back(std::make_tuple(
                    (size_t)ti,
                    BaryCoord<double>{
                        mesh_compute_.mesh().vertIndicesOfTriangle((size_t)ti),
                        {1.f - data.x - data.y, data.x, data.y},
                    },
                    float2(
                        (float)pos.x / (float)result.resolution().x * 2.f - 1.f,
                        1.f - (float)pos.y / (float)result.resolution().y * 2.f
                    )
                ));
                marking.valueAt(pos.x, pos.y) = (uint8_t)255;
            } else {
                marking.valueAt(pos.x, pos.y) = (uint8_t)50;
            }
        };

        // find the first pixel
        std::optional<int2> xy0;
        {
            // Timeit _("find first position");
            for (int y = 0; y < result.resolution().y; ++y)
            {
                for (int x = 0; x < result.resolution().x; ++x)
                {
                    const auto data = result.pixelAt<3>(x, y);
                    auto ti = _GetIndexInt(data.z, n_tri);
                    if (ti >= 0)
                    {
                        xy0 = int2((int)x, (int)y);
                        break;
                    }
                }
            }
        }

        if (xy0.has_value())
        {
            // Timeit _("find others");

            // (-1, -1)\ ( 0, -1) /( 1, -1)
            //          \    |   /
            // (-1,  0)<-( 0,  0)->( 1,  0)
            //          /    |   \
            // (-1,  1)/ ( 0,  1) \( 1,  1)
            static const std::vector<int2> s_goto_dirs = {
                { 1,  0}, { 1, -1}, { 0, -1}, {-1, -1},
                {-1,  0}, {-1,  1}, { 0,  1}, { 1,  1}
            };
            auto _nextDir = [=](const data::Image<float> & _img,
                                const int2 & _pos, size_t _i_last_dir) -> size_t
            {
                size_t i_from = (_i_last_dir + s_goto_dirs.size() / 2);
                for (size_t k = 1; k <= s_goto_dirs.size(); ++k)
                {
                    size_t i = (i_from + k) % s_goto_dirs.size();
                    int x = _pos.x + s_goto_dirs[i].x;
                    int y = _pos.y + s_goto_dirs[i].y;
                    if (0 <= x && x < _img.resolution().x &&
                        0 <= y && y < _img.resolution().y)
                    {
                        const auto data = _img.pixelAt<3>(x, y);
                        auto ti = _GetIndexInt(data.z, n_tri);
                        if (ti >= 0) { return i; }
                    }
                }
                // not found!
                return s_goto_dirs.size();
            };

            // first position
            int2 pos = xy0.value();
            _appendPossiblePosition(pos);
            const int n_pixels = marking.resolution().x * marking.resolution().y;

            // find all by linear search
            int n_visited_pixels = 0;
            size_t i_last_dir = 0;  // we scan by row, so the last dir is {1, 0}
            while (i_last_dir < s_goto_dirs.size())
            {
                i_last_dir = _nextDir(result, pos, i_last_dir);
                if (i_last_dir < s_goto_dirs.size())
                {
                    pos += s_goto_dirs[i_last_dir];
                    // break the loop in two conditions:
                    // 1) find the first pos agian
                    // 2) more then number of pixels (it's error)

                    if (pos == xy0.value()) { break; }
                    if (++n_visited_pixels > n_pixels) {
                        barycoords_on_contour.clear();
                        log::error("[contour]: loop over {} pixels, but only {} pixel in total. It must be wrong!",
                            n_visited_pixels, n_pixels);
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

        if (_debug_prefix.has_value() && (barycoords_on_contour.size() == 0 || _debug_must))
        {
            fs::create_directories(fs::path(_debug_prefix.value()).parent_path());
            result.save(_debug_prefix.value() + "_debug_render.hdr", true);
            marking.save(_debug_prefix.value() + "_cntr_mark.png", true);
        }

        return {barycoords_on_contour, marking};
    }
};

}

#endif
