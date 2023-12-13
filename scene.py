"""
This file contains the classes for the scene, geometry, material, and light.
"""
import numpy as np
from typing import Tuple


class BVHNode:
    """
    A node in the BVH tree. Each node has a bounding box, and either two children or a list of triangles.
    """
    def __init__(self):
        self.bbox_min = np.array([float('inf'), float('inf'), float('inf')])
        self.bbox_max = np.array([-float('inf'), -float('inf'), -float('inf')])
        self.left = None
        self.right = None
        self.triangles = []

    def is_leaf(self):
        return len(self.triangles) > 0
    
    def ray_intersects_bbox(self, ray_origin, ray_direction):
        t_min = np.empty(3)
        t_max = np.empty(3)
        
        for i in range(3):  # Iterate over x, y, z axes
            if ray_direction[i] == 0:
                t_min[i] = -float('inf')
                t_max[i] = float('inf')
                continue
            
            inv_dir = 1 / ray_direction[i]
            t_min[i] = (self.bbox_min[i] - ray_origin[i]) * inv_dir
            t_max[i] = (self.bbox_max[i] - ray_origin[i]) * inv_dir
            if inv_dir < 0:
                t_min[i], t_max[i] = t_max[i], t_min[i]
        
        # Calculate the furthest t_min and the closest t_max
        t_enter = t_min.max()
        t_exit = t_max.min()
        
        return t_enter <= t_exit and t_exit >= 0  # Intersects if t_enter <= t_exit and t_exit is positive

    @staticmethod
    def build_bvh(triangles: np.ndarray, depth=0):
        node = BVHNode()

        # Base case: if there's 1 or fewer triangles, it's a leaf node.
        if len(triangles) <= 1:
            if triangles:  # Check if there is at least one triangle
                node.bbox_min = triangles[0].min(axis=0)
                node.bbox_max = triangles[0].max(axis=0)
                node.triangles = triangles
            return node

        # Compute centroid bounding box for all triangles.
        centers = np.array([tri.mean(axis=0) for tri in triangles])
        centroid_bbox_min = centers.min(axis=0)
        centroid_bbox_max = centers.max(axis=0)

        # Find the axis along which to split: we choose the axis with the maximum extent.
        axis = np.argmax(centroid_bbox_max - centroid_bbox_min)

        # Find the median on the chosen axis.
        median = np.median(centers[:, axis])

        # Partition triangles into two sets: those less than the median and those greater or equal to the median.
        left_tris = [tri for i, tri in enumerate(triangles) if centers[i][axis] < median]
        right_tris = [tri for i, tri in enumerate(triangles) if centers[i][axis] >= median]

        # Handle the case where all centroids are at the exact same coordinate along the chosen axis
        # by splitting the list of triangles into two equal halves.
        if not left_tris or not right_tris:
            half_length = len(triangles) // 2
            left_tris, right_tris = triangles[:half_length], triangles[half_length:]

        # Recursive case: build the BVH for the two halves.
        node.left = BVHNode.build_bvh(left_tris, depth + 1)
        node.right = BVHNode.build_bvh(right_tris, depth + 1)

        # Update this node's bounding box to surround its children's bounding boxes.
        node.bbox_min = np.minimum(node.left.bbox_min if node.left else np.inf, node.right.bbox_min if node.right else np.inf)
        node.bbox_max = np.maximum(node.left.bbox_max if node.left else -np.inf, node.right.bbox_max if node.right else -np.inf)

        return node

class Light:
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity

class Material:
    def __init__(self, refraction_index, scattering_coefficients, color):
        self.refraction_index = refraction_index
        self.scattering_coefficients = scattering_coefficients
        self.color = color

class Scene:
    def __init__(self, vertices: np.ndarray, normals: np.ndarray, camera_pos: np.ndarray, camera_ori=None, use_sss=True):
        """
        Setup the scene with the geometry, materials, lights, and camera.

        Args:
            vertices: Numpy array of shape (N, 3, 3) representing the geometry's vertices.
            normals: Numpy array of shape (N, 3, 3) representing the normals at each vertex.

        Returns:
            A Scene object with the geometry, materials, lights, and camera set up.
        """
        self.geometry = []
        self.materials = []
        self.lights = []
        self.camera = {}
        self.bvh = None
        self.use_sss = use_sss

        # Normalize geometry to fit within the unit cube
        min_v = vertices.min(axis=(0, 1))
        max_v = vertices.max(axis=(0, 1))
        scale = 1 / max(max_v - min_v)

        normalized_vertices = (vertices - (min_v + max_v) / 2) * scale * 1.5  # Center the geometry at the origin
        
        # Add geometry
        self.add_geometry(normalized_vertices, normals)

        # Add material properties
        # Material should mimic a blue plastic
        material = Material(
            refraction_index=1.5,
            scattering_coefficients=np.array([0.1, 0.1, 0.1]),
            color=np.array([0.5, 0.5, 1.0])
        )

        self.add_material(material)

        # Add a light source (position, color, intensity)
        light = Light(np.array([2, 2, 2]), np.array([1.0, 1.0, 1.0]), 1.0)
        self.add_light(light)

        # Setup the camera
        # The camera is above the origin and shifted slightly to the left

        if camera_ori is None:    
            # Calculate the orientation of the camera to point at the origin
            target_point = np.array([0, 0, 0])  # You can adjust this if you want to look at a different point.
            world_up = np.array([0, 0, 1])  # Assuming Y is up in the world coordinate system.
            camera_ori = Scene.look_at(camera_pos, target_point, world_up)


        fov = 60  # Field of view in degrees
        self.set_camera(camera_pos, camera_ori, fov)

        self.print_scene()
        
        
    def add_geometry(self, vertices, normals):
        self.geometry.append({'vertices': vertices, 'normals': normals})
        triangles = [vertices[i] for i in range(vertices.shape[0])]
        self.bvh = BVHNode.build_bvh(triangles)
        
    def add_material(self, material):
        self.materials.append(material)
        
    def add_light(self, light):
        self.lights.append(light)
        
    def set_camera(self, position, orientation, fov):
        self.camera = {'position': position, 'orientation': orientation, 'fov': fov}

    def print_scene(self):
        """
        Print out basic info about the scene.
        """
        print("Scene Summary:")
        print("--------------")
        
        # Print geometries information
        print("Geometries:")
        for index, geometry in enumerate(self.geometry):
            vertices = geometry['vertices']
            bbox_min = vertices.min(axis=(0, 1))
            bbox_max = vertices.max(axis=(0, 1))
            print(f"  Geometry {index}:")
            print(f"    Bounding Box Min: {bbox_min}")
            print(f"    Bounding Box Max: {bbox_max}")
            print(f"    Triangle Count: {len(vertices)}")
        
        # Print materials information
        print("\nMaterials:")
        for index, material in enumerate(self.materials):
            print(f"  Material {index}:")
            print(f"    Refraction Index: {material.refraction_index}")
            print(f"    Scattering Coefficients: {material.scattering_coefficients}")
            print(f"    Color: {material.color}")
        
        # Print lights information
        print("\nLights:")
        for index, light in enumerate(self.lights):
            print(f"  Light {index}:")
            print(f"    Position: {light.position}")
            print(f"    Color: {light.color}")
            print(f"    Intensity: {light.intensity}")
        
        # Print camera information
        print("\nCamera:")
        camera = self.camera
        print(f"  Position: {camera['position']}")
        print(f"  Orientation:\n{camera['orientation']}")
        print(f"  Field of View: {camera['fov']} degrees")

        print("\n--------------")
        print("End of Scene Summary\n")

    @staticmethod
    def ray_intersects_triangle(ray_origin: np.array, ray_direction: np.array, triangle_vertices: Tuple[np.array]) -> Tuple[bool, float]:
        """Möller-Trumbore intersection algorithm"""
        v0, v1, v2 = triangle_vertices
        edge1, edge2 = v1 - v0, v2 - v0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)
        if -1e-7 < a < 1e-7:
            return False, None  # This means the ray is parallel to the triangle.
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False, None
        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)
        if v < 0.0 or u + v > 1.0:
            return False, None
        # At this stage we can compute t to find out where the intersection point is on the line.
        t = f * np.dot(edge2, q)
        if t > 1e-7:  # ray intersection
            return True, t
        else:  # This means that there is a line intersection but not a ray intersection.
            return False, None
            
    def ray_march(self, ray_origin: np.array, ray_direction: np.array, max_distance: float, epsilon: float) -> Tuple[bool, np.array, np.array]:
        closest_intersection = max_distance
        intersection_found = False
        intersection_point = None
        normal_at_intersection = None

        # Use a stack instead of the recursive function to traverse the BVH tree
        stack = [self.bvh]
        while stack:
            node = stack.pop()
            if node is None or not node.ray_intersects_bbox(ray_origin, ray_direction):
                continue

            if node.is_leaf():
                for triangle in node.triangles:
                    has_intersection, distance = self.ray_intersects_triangle(ray_origin, ray_direction, triangle)
                    if has_intersection and distance < closest_intersection:
                        intersection_found = True
                        closest_intersection = distance
                        intersection_point = ray_origin + distance * ray_direction
                        normal_at_intersection = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                        normal_at_intersection /= np.linalg.norm(normal_at_intersection)
            else:
                # Add child nodes to the stack
                stack.append(node.right)
                stack.append(node.left)

        return intersection_found, intersection_point, normal_at_intersection
        
    @staticmethod
    def look_at(camera_pos, target, up):
        # Calculate forward vector (direction the camera is looking)
        forward = np.array(target) - np.array(camera_pos)
        forward /= np.linalg.norm(forward)
        
        # Calculate right vector
        right = np.cross(forward, np.array(up))
        right /= np.linalg.norm(right)

        # Calculate true up vector
        true_up = np.cross(right, forward)

        # Construct a 3x3 orientation matrix (camera to world basis transform)
        orientation = np.stack((right, true_up, -forward), axis=-1)
        return orientation
