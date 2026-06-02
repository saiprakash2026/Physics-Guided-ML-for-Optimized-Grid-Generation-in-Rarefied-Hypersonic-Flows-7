import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Point, Polygon


# ===============================================================
# (A) Local-neighborhood lambda propagation (manual buffer)
# ===============================================================
def build_length_field_local_min(coords, raw_lengths,
                                 r_buffer=0.1,
                                 smooth_iter=3,
                                 relax=0.3):

    tree = cKDTree(coords)
    L = raw_lengths.copy()

    for i, pt in enumerate(coords):
        idx = tree.query_ball_point(pt, r_buffer)
        if idx:
            L[i] = min(L[i], np.min(raw_lengths[idx]))

    logL = np.log(L)
    for _ in range(smooth_iter):
        new_logL = logL.copy()
        for i, pt in enumerate(coords):
            _, idx = tree.query(pt, k=12)
            avg = np.mean(logL[idx])
            new_logL[i] = (1.0 - relax) * logL[i] + relax * avg
        logL = new_logL

    return np.exp(logL)


# ===============================================================
# (B) Read input density file
# ===============================================================
def read_region_thresholds(filename, lambda_cap=0.001):
    data = np.loadtxt(filename)
    coords = data[:, :2]
    lambdas = np.maximum(data[:, 2], lambda_cap)
    return coords, lambdas


# ===============================================================
# (C) Assign cell size from lambda (LOG-COMPRESSED)
# ===============================================================
def assign_length_from_lambda(coords, lambdas,
                              target_ratio=0.3,
                              blend_width=0.05,
                              buffer_radius=0.1,
                              alpha=0.35,
                              h_min=0.02,
                              h_max=0.4):

    # Reference lambda (robust)
    lambda_ref = np.median(lambdas)

    # Log-compressed mapping
    raw_lengths = (
        target_ratio * lambda_ref *
        (lambdas / lambda_ref) ** alpha
    )

    # Absolute safety bounds
    raw_lengths = np.clip(raw_lengths, h_min, h_max)

    length_field = build_length_field_local_min(
        coords=coords,
        raw_lengths=raw_lengths,
        r_buffer=buffer_radius
    )

    tree = cKDTree(coords)

    def get_target_length(pt):
        d, idx = tree.query(pt, k=5)
        w = 1.0 / (d**2 + blend_width**2)
        w /= np.sum(w)
        return np.sum(w * length_field[idx])

    return get_target_length


# ===============================================================
# (D) Adaptive refinement (strict Delaunay)
# ===============================================================
def refine_mesh_by_edge_length(points, coords, lambdas,
                               target_ratio=0.3,
                               max_iter=10,
                               blend_width=0.05,
                               buffer_radius=0.1,
                               alpha=0.35):

    pts = points.copy()

    get_length = assign_length_from_lambda(
        coords, lambdas,
        target_ratio=target_ratio,
        blend_width=blend_width,
        buffer_radius=buffer_radius,
        alpha=alpha
    )

    for it in range(max_iter):
        tri = Delaunay(pts)
        new_pts = []
        edges_seen = set()

        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i+1) % 3]
                key = tuple(sorted((a, b)))
                if key in edges_seen:
                    continue
                edges_seen.add(key)

                v1, v2 = pts[a], pts[b]
                mid = 0.5 * (v1 + v2)
                elen = np.linalg.norm(v2 - v1)

                target = get_length(mid)

                if elen > 1.15 * target:
                    if np.min(np.linalg.norm(pts - mid, axis=1)) > 1e-6:
                        new_pts.append(mid)

        if not new_pts:
            print(f"[Refinement] Iter {it}: converged.")
            break

        new_pts = np.unique(np.array(new_pts), axis=0)
        pts = np.vstack([pts, new_pts])
        print(f"[Refinement] Iter {it}: +{len(new_pts)} pts → {len(pts)} total")

    return pts


# ===============================================================
# (E) Geometry utilities
# ===============================================================
def create_initial_grid(bounds, spacing):
    x_min, x_max, y_min, y_max = bounds
    x = np.arange(x_min, x_max, spacing)
    y = np.arange(y_min, y_max, spacing)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def create_bluff_body_adaptive(center, diameter, get_length):
    cx, cy = center
    r = diameter * 0.5

    th = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    fine_pts = np.column_stack([cx + r*np.cos(th), cy + r*np.sin(th)])

    keep = [0]
    acc = 0.0

    for i in range(1, len(fine_pts)):
        ds = np.linalg.norm(fine_pts[i] - fine_pts[i-1])
        acc += ds

        normal = fine_pts[i] - np.array([cx, cy])
        normal /= np.linalg.norm(normal)
        probe = fine_pts[i] + 0.01 * normal

        h = get_length(probe)

        if acc >= h:
            keep.append(i)
            acc = 0.0

    bluff_pts = fine_pts[keep]
    return bluff_pts, Polygon(bluff_pts)


def add_domain_boundaries(bounds, spacing=0.05):
    x_min, x_max, y_min, y_max = bounds
    xr = np.arange(x_min, x_max + spacing, spacing)
    yr = np.arange(y_min, y_max + spacing, spacing)
    inlet  = np.column_stack([np.full_like(yr, x_min), yr])
    outlet = np.column_stack([np.full_like(yr, x_max), yr])
    top    = np.column_stack([xr, np.full_like(xr, y_max)])
    bottom = np.column_stack([xr, np.full_like(xr, y_min)])
    return inlet, outlet, top, bottom


# ===============================================================
# (F) Mesh filtering & classification
# ===============================================================
def filter_excluded_triangles(tri, exclusion_zone):
    valid = []
    for t in tri.simplices:
        c = np.mean(tri.points[t], axis=0)
        if not exclusion_zone.contains(Point(c)):
            valid.append(t)
    return np.array(valid)


def classify_boundary_edges(points, triangles, segments, tol=0.05):
    edge_count = {}
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted((tri[i], tri[(i+1) % 3])))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    out = {k: [] for k in segments}
    for e in boundary_edges:
        mid = points[list(e)].mean(axis=0)
        for name, seg in segments.items():
            if np.min(np.linalg.norm(seg - mid, axis=1)) < tol:
                out[name].append(e)
                break
    return out


# ===============================================================
# (G) Fluent writer
# ===============================================================
def write_fluent(filename, points2d, triangles, boundaries, height):
    n = len(points2d)
    total_faces = sum(len(v) for v in boundaries.values())
    total_prisms = len(triangles)

    with open(filename, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        f.write(f"$PhysicalNames\n{1 + len(boundaries)}\n")
        f.write("2 1 \"fluid\"\n")
        for i, name in enumerate(boundaries.keys(), 2):
            f.write(f"1 {i} \"{name}\"\n")
        f.write("$EndPhysicalNames\n")

        f.write("$Nodes\n")
        f.write(f"{2*n}\n")
        for i, (x, y) in enumerate(points2d):
            f.write(f"{i+1} {x} {y} 0.0\n")
        for i, (x, y) in enumerate(points2d):
            f.write(f"{i+1+n} {x} {y} {height}\n")
        f.write("$EndNodes\n")

        f.write("$Elements\n")
        f.write(f"{total_prisms + total_faces}\n")

        for i, tri in enumerate(triangles):
            a, b, c = tri + 1
            f.write(f"{i+1} 6 2 1 1 {a} {b} {c} {a+n} {b+n} {c+n}\n")

        eid = total_prisms + 1
        for phys_id, (name, edges) in enumerate(boundaries.items(), 2):
            for e in edges:
                a, b = e
                a += 1
                b += 1
                f.write(f"{eid} 3 2 {phys_id} {phys_id} {a} {b} {b+n} {a+n}\n")
                eid += 1

        f.write("$EndElements\n")


# ===============================================================
# (H) Main mesh generator
# ===============================================================
def generate_mesh(config):

    coords, lambdas = read_region_thresholds(
        config["input_file"],
        lambda_cap=config.get("lambda_cap", 0.001)
    )

    initial_pts = create_initial_grid(
        config["domain_bounds"],
        config["initial_spacing"]
    )

    refined_pts = refine_mesh_by_edge_length(
        initial_pts, coords, lambdas,
        target_ratio=config.get("target_ratio", 0.3),
        max_iter=config["max_iter"],
        buffer_radius=config.get("buffer_radius", 0.1),
        alpha=config.get("alpha", 0.35)
    )

    get_length = assign_length_from_lambda(
        coords, lambdas,
        target_ratio=config.get("target_ratio", 0.3),
        buffer_radius=config.get("buffer_radius", 0.1),
        alpha=config.get("alpha", 0.35)
    )

    bluff_pts, exclusion = create_bluff_body_adaptive(
        config["bluff_body_center"],
        config["bluff_body_size"],
        get_length
    )

    refined_pts = np.array([
        p for p in refined_pts
        if not exclusion.contains(Point(p))
    ])

    inlet, outlet, top, bottom = add_domain_boundaries(config["domain_bounds"])
    all_pts = np.vstack([refined_pts, bluff_pts, inlet, outlet, top, bottom])

    tri = Delaunay(all_pts)
    tris = filter_excluded_triangles(tri, exclusion)

    segments = {
        "inlet": inlet,
        "outlet": outlet,
        "top": top,
        "bottom": bottom,
        "wall": bluff_pts
    }

    boundaries = classify_boundary_edges(all_pts, tris, segments)

    write_fluent(
        config["output_file"],
        all_pts, tris, boundaries,
        config["extrusion_height"]
    )


# ===============================================================
# RUN BLOCK
# ===============================================================
if __name__ == "__main__":

    config = {
        "input_file": "numb_density_data.txt",
        "output_file": "bird_condition_mesh.msh",

        "bluff_body_center": (0.5, 0),
        "bluff_body_size": 1.0,

        "initial_spacing": 0.2,
        "max_iter": 15,

        "extrusion_height": 1.0,
        "domain_bounds": (-4, 9, -4.5, 4.5),

        "target_ratio": 100.00,
        "lambda_cap": 0.00008,

        "buffer_radius": 0.2,
        "alpha": 0.25
    }

    generate_mesh(config)
