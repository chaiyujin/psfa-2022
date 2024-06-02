#define _side(p0, p1, p2)  \
    ( (((p1)[0] - (p0)[0])*((p2)[1] - (p1)[1]) - ((p2)[0] - (p1)[0])*((p1)[1] - (p0)[1])) >= 0 )


__device__ bool _is_in_polygon(const float *xy, const float *pts, int nb_pts)
{
    bool same = true;
    bool r_last = false;
    int pj, i, j;
	for (int pi = 0; pi < nb_pts; ++pi)
	{
        // get indices
        pj = pi + 1; if (pj == nb_pts) pj = 0;
        i = pi * 2;
        j = pj * 2;
        // judge side
		bool r = _side(xy, &pts[i], &pts[j]);
		if (pi == 0) r_last = r;
        else if (r_last != r)
        {
            same = false;
            break;
        }
	}
	return same;
}

__global__ void copy_image(
    float       * out,
    const float * img,
    const float * trimesh,
    const int   * img_shape,
    const int   * mesh_shape
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = img_shape[0];
    const int w = img_shape[1];
    const int b = img_shape[2];
    if (x < w && y < h)
    {
        const int i = (y * w + x) * b;
        const int nb_tri = mesh_shape[0];
        const float xy[2] = {(float)x, (float)y};
        for (int it = 0; it < nb_tri; ++it)
        {
            bool in = _is_in_polygon(xy, &trimesh[it*6], 3);
            if (in)
            {
                for (int c = 0; c < b; ++c)
                {
                    out[i+c] = img[i+c];
                }
                break;
            }
        }
    }
}


__global__ void face_mask(
    float       * masks,
    float       * full_mask,
    const float * trimesh,
    const int   * out_shape
) {
    const int nb_tri = out_shape[0];
    const int h = out_shape[1];
    const int w = out_shape[2];
    const int b = out_shape[3];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride = h * w * b;
    const int pi = (y * w + x) * b;
    if (x < w && y < h)  // pixel in image
    {
        // clear first
        for (int it = 0, i = pi; it < nb_tri; ++it, i += stride)
        {
            for (int c = 0; c < b; ++c)
            {
                masks[i+c] = 0.0;
            }
        }
        for (int c = 0; c < b; ++c)
        {
            full_mask[pi+c] = 0.0;
        }

        // try to find
        const float xy[2] = {(float)x, (float)y};
        for (int it = 0, i = pi; it < nb_tri; ++it, i += stride)
        {
            bool in = _is_in_polygon(xy, &trimesh[it*6], 3);
            if (in)
            {
                for (int c = 0; c < b; ++c)
                {
                    masks[i+c] = 1.0;
                    full_mask[pi+c] = 1.0;
                }
                break;
            }
        }
    }
}


__global__ void warp_image(
    float       * out,
    const float * img,
    const float * trimesh,
    const float * tforms,
    const int   * out_shape,
    const int   * img_shape,
    const int   * mesh_shape
) {
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    const int ho = out_shape[0];
    const int wo = out_shape[1];
    const int bo = out_shape[2];
    const int hi = img_shape[0];
    const int wi = img_shape[1];
    const int bi = img_shape[2];
    if (xo < wo && yo < ho)
    {
        const int io = (yo * wo + xo) * bo;
        const int nb_tri = mesh_shape[0];
        const float xy[2] = {(float)xo, (float)yo};
        bool found = false;
        for (int it = 0; it < nb_tri; ++it)
        {
            bool in = _is_in_polygon(xy, &trimesh[it*6], 3);
            if (in)
            {
                // get the coordinates at input image
                const int imat = it * 6;
                const float xi = tforms[imat+0] * xo + tforms[imat+1] * yo + tforms[imat+2];
                const float yi = tforms[imat+3] * xo + tforms[imat+4] * yo + tforms[imat+5];
                // bilinear
                const int x0 = (int)xi, y0 = (int)yi;
                if (0 <= x0 && x0 < wi && 0 <= y0 && y0 < hi)
                {
                    int x1 = (x0 + 1 == wi) ? x0 : (x0 + 1);
                    int y1 = (y0 + 1 == hi) ? y0 : (y0 + 1);
                    const float ax = xi - x0;
                    const float ay = yi - y0;
                    const float a00 = (1-ax)*(1-ay);
                    const float a01 = (1-ax)*ay;
                    const float a10 = ax*(1-ay);
                    const float a11 = ax*ay;
                    const int i00 = (x0 + y0 * wi) * bi;
                    const int i01 = (x0 + y1 * wi) * bi;
                    const int i10 = (x1 + y0 * wi) * bi;
                    const int i11 = (x1 + y1 * wi) * bi;
                    for (int c = 0; c < bo; ++c)
                    {
                        if (c < bi)
                            out[io+c] = (
                                img[i00+c] * a00 +
                                img[i01+c] * a01 +
                                img[i10+c] * a10 +
                                img[i11+c] * a11
                            );
                        else
                            out[io+c] = 1.0;
                    }
                }
                found = true;
                break;
            }
        }
        if (!found)
        {
            for (int c = 0; c < bo; ++c)
            {
                out[io+c] = 0.0;
            }
        }
    }
}
