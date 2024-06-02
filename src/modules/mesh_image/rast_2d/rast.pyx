import numpy

cimport cython
cimport numpy as np

DTYPE = numpy.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _sign(np.ndarray p1, np.ndarray p2, np.ndarray p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

@cython.boundscheck(False)
@cython.wraparound(False)
def point_in_triangle(
    np.ndarray point,
    np.ndarray tri_pts
):
    cdef float minx = min(tri_pts[:, 0])
    cdef float maxx = max(tri_pts[:, 0])
    if (point[0] < minx or point[0] > maxx): return False
    cdef float miny = min(tri_pts[:, 1])
    cdef float maxy = max(tri_pts[:, 1])
    if (point[1] < miny or point[1] > maxy): return False

    cdef float d1 = _sign(point, tri_pts[0], tri_pts[1])
    cdef float d2 = _sign(point, tri_pts[1], tri_pts[2])
    cdef float d3 = _sign(point, tri_pts[2], tri_pts[0])

    cdef int has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    cdef int has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def barycenteric_coords(point, vertices):
    C, B, A = vertices
    P = point

    # Compute vectors
    v0 = C - A
    v1 = B - A
    v2 = P - A

    # Compute dot products
    dot00 = numpy.dot(v0, v0)
    dot01 = numpy.dot(v0, v1)
    dot02 = numpy.dot(v0, v2)
    dot11 = numpy.dot(v1, v1)
    dot12 = numpy.dot(v1, v2)

    # Compute barycentric coordinates
    denom = (dot00 * dot11 - dot01 * dot01)
    if numpy.abs(denom) < 1e-5:
        denom = numpy.sign(denom) * 1e-5
    invDenom = 1 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return u, v


@cython.boundscheck(False)
@cython.wraparound(False)
def rasterize_triangles_2d(np.ndarray pts, np.ndarray triangles, int h, int w):
    cdef float minx = min(pts[:, 0])
    cdef float maxx = max(pts[:, 0])
    cdef float miny = min(pts[:, 1])
    cdef float maxy = max(pts[:, 1])

    point     = numpy.zeros((2), DTYPE)
    tri_pts   = numpy.zeros((len(triangles), 3, 2), DTYPE)
    face_mask = numpy.zeros((h, w, 2), DTYPE)
    rast_out  = numpy.zeros((h, w, 4), DTYPE)

    cdef int last_hit = -1
    cdef int found = 0

    for i_tri in range(len(triangles)):
        tri = triangles[i_tri]
        tri_pts[i_tri, 0] = pts[tri[0]]
        tri_pts[i_tri, 1] = pts[tri[1]]
        tri_pts[i_tri, 2] = pts[tri[2]]

    pixel = 0
    all_pixels = h * w

    for y in range(h):
        for x in range(w):
            # large bbox
            if not (x < minx or x > maxx or y < miny or y > maxy):
                found = 0
                point[0] = x
                point[1] = y
                if last_hit >= 0:
                    i_tri = last_hit
                    u, v = barycenteric_coords(point, tri_pts[i_tri])
                    if u >= 0 and v >= 0 and u+v<=1:
                        rast_out[y, x, 0] = u
                        rast_out[y, x, 1] = v
                        rast_out[y, x, 2] = 0.0
                        rast_out[y, x, 3] = i_tri + 1
                        found = 1

                if found == 0:
                    for i_tri in range(len(triangles)):
                        if i_tri == last_hit:
                            continue
                        tri = triangles[i_tri]
                        u, v = barycenteric_coords(point, tri_pts[i_tri])
                        if u >= 0 and v >= 0 and u+v<=1:
                            rast_out[y, x, 0] = u
                            rast_out[y, x, 1] = v
                            rast_out[y, x, 2] = 0.0
                            rast_out[y, x, 3] = i_tri + 1
                            found = 1
                            last_hit = i_tri
                            break
                if found == 1:
                    face_mask[y, x] = 1.0

            pixel += 1
            print("generate_masks %.1f%%" % (pixel * 100 / all_pixels), end='\r')

    return rast_out
