"""
This file handles the actual subsurface scattering simulation through ray tracing.
"""

import ctypes
from multiprocessing import Pool, Array
import os
import numpy as np
from tqdm import tqdm
from numba import jit, float32, vectorize

from scene import Scene

class Renderer:
    def __init__(self):
        pass

    @staticmethod
    @jit(forceobj=True, fastmath=True)
    def calculate_color(scene: Scene, intersection_point: np.array, normal_at_intersection: np.array) -> np.array:
        """
        Extend the function to combine both the Phong lighting and SSS contributions.
        """
        # Calculate the regular Phong lighting color
        phong_color = Renderer.calculate_phong_color(scene, intersection_point, normal_at_intersection) 
        
        if scene.use_sss:
            # Calculate the subsurface scattering color
            sss_color = Renderer.calculate_subsurface_color(scene, intersection_point, normal_at_intersection)
        
            # Mix the Phong color with the SSS color based on a weighted factor
            # This weight can be part of the material properties to indicate how strong the SSS effect is for this material
            sss_weight = 0.3  # Adjust this factor to achieve the desired effect
            combined_color = (1 - sss_weight) * phong_color + sss_weight * sss_color
            
            # Make sure we still respect the 0-255 range
            combined_color = np.clip(combined_color, 0, 255).astype(np.uint8)
        
            return combined_color
        else:
            return phong_color

    @staticmethod
    @jit(forceobj=True, fastmath=True)
    def calculate_phong_color(scene: Scene, intersection_point: np.array, normal_at_intersection: np.array) -> np.array:
        """
        This function calculates the color at the given intersection point using the Phong reflection model
        with color values in the range 0 to 255.
        
        Args:
            scene: The scene object.
            intersection_point: The intersection point.
            normal_at_intersection: The normal at the intersection point.

        Returns:
            The color at the intersection point as an array of integers in the range 0 to 255.
        """ 
        # Ambient lighting - static ambient light level
        ambient_strength = 0.1
        ambient_color = np.array([255, 255, 255])  # White ambient light
        
        # Assuming there is one material and one light source in the scene
        material = scene.materials[0]
        light = scene.lights[0]

        # Convert material and light color to 0-255 scale and ensure they are np.array of integers
        material_color = np.array(material.color * 255, dtype=int)
        light_color = np.array(light.color * 255, dtype=int)

        # The resulting color starts with the ambient light
        resulting_color = material_color * ambient_strength * (ambient_color / 255)

        # Diffuse lighting
        light_dir = light.position - intersection_point
        light_dir /= np.linalg.norm(light_dir)  # Normalize light direction
        diff = max(np.dot(normal_at_intersection, light_dir), 0.0)
        diffuse = diff * material_color * (light_color / 255) * light.intensity

        # Specular lighting
        viewer_dir = scene.camera['position'] - intersection_point
        viewer_dir /= np.linalg.norm(viewer_dir)  # Normalize viewer direction
        reflect_dir = 2 * normal_at_intersection * np.dot(normal_at_intersection, light_dir) - light_dir
        reflect_dir /= np.linalg.norm(reflect_dir)  # Normalize reflection direction

        specular_strength = 0.5
        shininess = 32
        spec = pow(max(np.dot(viewer_dir, reflect_dir), 0.0), shininess)
        specular = specular_strength * spec * (light_color / 255) * light.intensity

        # Combine the ambient, diffuse, and specular components
        resulting_color += diffuse + specular
        
        # Ensure that the color does not exceed the maximum value of 255 in any channel
        resulting_color = np.clip(resulting_color, 0, 255)
        
        # Convert the result to integers as we're dealing with discrete color values
        return resulting_color.astype(np.uint8)

    @staticmethod
    @jit(forceobj=True, fastmath=True)
    def calculate_subsurface_color(scene: Scene, intersection_point: np.array, normal_at_intersection: np.array) -> np.array:
        """
        This function simulates the subsurface scattering effect at a given
        intersection point using a simple BSSRDF model.
        """
        # Assuming there's just one material in the scene for simplification
        material = scene.materials[0]
        light = scene.lights[0]  # Assuming one direct light source for simplicity

        # Constants and scale factors for the simple BSSRDF model
        scattering_radius = 0.1  # Adjust this value according to the material's scattering properties
        samples_count = 25  # The number of samples to evaluate the BSSRDF
        falloff = 1.0 / (4 * np.pi * scattering_radius**2)  # Simplistic falloff factor based on the scattering radius

        sss_color = np.zeros(3)
        
        # Retrieve the material and light color, convert to the range 0 to 1
        light_color = light.color
        material_color = material.color

        # Calculate the direction to the light source
        light_dir = light.position - intersection_point
        light_distance = np.linalg.norm(light_dir)
        light_dir /= light_distance  # Normalize
        light_intensity = light.intensity / (light_distance**2)  # Attenuate based on distance squared

        # Find the light's contribution to the subsurface scattering
        light_contribution = light_color * light_intensity

        # Iterate over sample points to simulate the BSSRDF
        for i in range(samples_count):
            # Generate a sample point within the scattering radius around the point of intersection
            offset = np.random.uniform(-1, 1, 3)
            offset /= np.linalg.norm(offset)
            offset *= scattering_radius * np.sqrt(np.random.uniform(0, 1))  # Ensure uniform distribution over the sphere

            # Sample point is the offset from the intersection point inside the material
            sample_point = intersection_point + offset
            
            # Calculate the distance between the sample point and the original intersection point
            r = np.linalg.norm(sample_point - intersection_point)
            
            # Rd(r) is the BSSRDF's diffuse reflectance, which diminishes with the square of the distance 
            Rd = falloff * np.exp(-material.scattering_coefficients * r) / r**2
            
            # Accumulate the weighted subsurface scattering color
            sss_color += Rd * material_color * light_contribution
        
        # Divide by the number of samples to normalize the SSS contribution
        sss_color /= samples_count
        
        # Finally, multiply by 255 to adapt the range from [0, 1] to [0, 255] and clamp to valid color values
        sss_color *= 255
        sss_color = np.clip(sss_color, 0, 255)

        return sss_color.astype(np.uint8)
        
    @staticmethod
    @jit(forceobj=True, fastmath=True)
    def raytrace(scene: Scene, pixel: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        This function performs ray tracing for a single pixel, using the
        ray_march function to find the intersection point and the normal
        at that point.
        """
        # Extract the camera parameters
        cam_pos = scene.camera['position']
        cam_ori_matrix = scene.camera['orientation']  # Now a 3x3 matrix
        fov = scene.camera['fov']

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate scales for x and y based on fov
        scale_x = np.tan(np.radians(fov) / 2) * aspect_ratio
        scale_y = np.tan(np.radians(fov) / 2)
        
        # Map pixel to [-1, 1] range (NDC space)
        x_ndc = (2 * (pixel[0] / width) - 1) * scale_x
        y_ndc = (1 - 2 * (pixel[1] / height)) * scale_y
        
        # Calculate the direction vector (in camera space) based on NDC space
        direction_camera_space = np.array([x_ndc, y_ndc, -1], dtype=np.float32)  # z-coordinate is -1 because the camera looks towards negative z.
        direction_camera_space /= np.linalg.norm(direction_camera_space)
        
        # Transform the direction from camera space to world space using the camera orientation matrix
        ray_direction_world_space = cam_ori_matrix @ direction_camera_space
        ray_direction_world_space /= np.linalg.norm(ray_direction_world_space)

        # Perform ray marching to find the intersection point and normal
        intersection_found, intersection_point, normal_at_intersection = scene.ray_march(cam_pos, ray_direction_world_space, 3, 0.00001)
        
        if intersection_found:
            # Calculate the color at the intersection point
            return Renderer.calculate_color(scene, intersection_point, normal_at_intersection)
        else:
            # No intersection found, return background color (e.g., black)
            return np.array([0, 0, 0])

    @staticmethod
    def raytrace_chunk(args):
        scene, y_row, width, height, samples, = args
        # Access the global shared memory array within this worker.
        global shared_output_array
        for x in range(width):
            # Sample and raytrace code here.
            color = Renderer.raytrace(scene, np.array([x, y_row]), width, height)
            shared_output_array[y_row, x] = color

    @staticmethod
    def init_pool(shared_array_base, width, height):
        # Initialize the shared memory buffer to be accessible to each pool worker.
        global shared_output_array
        shared_output_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_output_array = shared_output_array.reshape(height, width, 3)
    
    @staticmethod
    def render(scene: Scene, width: int, height: int, samples: int) -> np.ndarray:
        # Create a shared ctypes array with an appropriate lock
        shared_array_base = Array(ctypes.c_uint8, width * height * 3)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(height, width, 3)
        
        THREADS = os.cpu_count()
        pool = Pool(THREADS, initializer=Renderer.init_pool, initargs=(shared_array_base, width, height,))

        # Create a tqdm progress bar
        pbar = tqdm(total=height)

        # Define tasks based on rows to distribute to processes.
        tasks = [(scene, y, width, height, samples) for y in range(height)]
        
        # Keep track of the result handles
        result_handles = [pool.apply_async(Renderer.raytrace_chunk, args=(task,), callback=lambda _: pbar.update(1)) for task in tasks]

        # Prevent the main process from continuing until all results are done
        for result in result_handles:
            result.wait()

        # Close the pool and wait for all workers to finish
        pool.close()
        pool.join()

        # Ensure the progress bar is closed
        pbar.close()

        return shared_array