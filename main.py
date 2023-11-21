"""
This is the main file for the subsurface scattering project.
It mainly handles argument parsing and calling the appropriate functions.
"""

import argparse

import numpy as np
from load_stl import load_stl
from scene import Scene
from render import Renderer
from save_image import save_image

if __name__ == "__main__":
    # Make ctrl-c force quit
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)


    # Argument parsing
    parser = argparse.ArgumentParser(description="Subsurface scattering simulation")
    parser.add_argument("input", help="The input STL file")
    parser.add_argument("output", help="The output PNG file")
    parser.add_argument("--width", type=int, default=80, help="The width of the output image")
    parser.add_argument("--height", type=int, default=45, help="The height of the output image")
    parser.add_argument("--samples", type=int, default=100, help="The number of samples to take per pixel")
    parser.add_argument("--position", type=float, nargs=3, default=[0, 1, 1], help="The position of the camera")

    args = parser.parse_args()

    # Convert the position to a numpy array
    args.position = np.array(args.position, dtype=np.float32)

    # Load the STL file
    print("Loading STL file...")
    vertices, normals = load_stl(args.input)

    print("Setting up render...")
    scene = Scene(vertices, normals, args.position)

    print("Rendering image...")
    processed_data = Renderer.render(scene, args.width, args.height, args.samples)
    
    print("Saving image...")
    save_image(args.output, processed_data)