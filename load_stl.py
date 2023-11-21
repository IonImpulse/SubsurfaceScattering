"""
This file handles loading of STL files into the program.
It returns a numpy array of the vertices and a numpy array of the normals.
"""

import numpy as np
import os
import struct

def load_stl(path: str) -> (np.ndarray, np.ndarray):
    def read_triangle(file):
        normal = struct.unpack('3f', file.read(12))
        vertex1 = struct.unpack('3f', file.read(12))
        vertex2 = struct.unpack('3f', file.read(12))
        vertex3 = struct.unpack('3f', file.read(12))
        file.read(2)  # skip attribute byte count
        return np.array(normal), np.array([vertex1, vertex2, vertex3])

    def read_binary_stl(file):
        file.read(80)  # skip header
        num_triangles = struct.unpack('I', file.read(4))[0]
        normals = np.zeros((num_triangles, 3), dtype=np.float32)
        vertices = np.zeros((num_triangles, 3, 3), dtype=np.float32)
        for i in range(num_triangles):
            normals[i], vertices[i] = read_triangle(file)
        return vertices, normals
    
    def read_ascii_stl(file):
        normals = []
        vertices = []
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'facet' and parts[1] == 'normal':
                normals.append(np.array(parts[-3:], dtype=float))
            elif parts[0] == 'vertex':
                vertices.append(np.array(parts[-3:], dtype=float))
        
        normals = np.array(normals, dtype=np.float32)
        vertices = np.array(vertices, dtype=np.float32).reshape((-1, 3, 3))

        return vertices, normals

    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    with open(path, 'r') as file:  # Open as text for ASCII parsing
        try:
            first_line = file.readline()
            file.seek(0)  # reset to beginning of the file
            if 'solid' in first_line:
                # it's an ASCII STL
                return read_ascii_stl(file)
        except Exception as e:
            pass
        
        # it's a binary STL, re-open as binary
        with open(path, 'rb') as binary_file:
            return read_binary_stl(binary_file)
