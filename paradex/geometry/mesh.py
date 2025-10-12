import numpy as np

def point_in_convex_hull(points, mesh_vertices):
    """
    Check if points are inside a convex mesh using ConvexHull
    Works well for convex meshes
    """
    try:
        hull = ConvexHull(mesh_vertices)
        
        # For each point, check if it's inside the hull
        inside_points = []
        for point in points:
            # Check if point satisfies all hull constraints
            is_inside = True
            for eq in hull.equations:
                if np.dot(eq[:-1], point) + eq[-1] > 1e-10:  # Small tolerance
                    is_inside = False
                    break
            inside_points.append(is_inside)
        
        return np.array(inside_points)
    except Exception as e:
        print(f"ConvexHull failed: {e}")
        return np.zeros(len(points), dtype=bool)
