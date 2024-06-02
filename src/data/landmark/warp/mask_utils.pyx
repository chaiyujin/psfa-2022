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


@cython.boundscheck(False)
@cython.wraparound(False)
def generate_masks(np.ndarray pts, np.ndarray triangles, int h, int w):
    cdef float minx = min(pts[:, 0])
    cdef float maxx = max(pts[:, 0])
    cdef float miny = min(pts[:, 1])
    cdef float maxy = max(pts[:, 1])

    point = numpy.zeros((2), DTYPE)
    tri_pts = numpy.zeros((len(triangles), 3, 2), DTYPE)
    face_mask = numpy.zeros((h, w, 2), DTYPE)
    triangle_masks = numpy.zeros((len(triangles), h, w, 2), DTYPE)

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
                    if point_in_triangle(point, tri_pts[i_tri]):
                        triangle_masks[i_tri, y, x] = 1.0
                        found = 1

                if found == 0:
                    for i_tri in range(len(triangles)):
                        if i_tri == last_hit:
                            continue
                        tri = triangles[i_tri]
                        if point_in_triangle(point, tri_pts[i_tri]):
                            triangle_masks[i_tri, y, x] = 1.0
                            found = 1
                            last_hit = i_tri
                            break
                if found == 1:
                    face_mask[y, x] = 1.0

            pixel += 1
            print("generate_masks %.1f%%" % (pixel * 100 / all_pixels), end='\r')

    return triangle_masks, face_mask
