# sfm_pipeline/image_loader.py

import os
from typing import List

import cv2
import numpy as np

class ImgLoader:
    """
    Image Loader class responsible for loading images from a directory,
    reading and scaling the associated camera intrinsic matrix,
    and providing functionality to downscale images.
    """
    
    def __init__(self, img_dir: str, scale_factor: float):
        """
        Initializes the image loader with the directory containing images
        and a scaling factor for resizing images and camera matrix.
        
        Parameters:
        - img_dir: Path to the directory containing images and camera matrix file.
        - scale_factor: Factor by which images and intrinsic matrix will be scaled down.
        """
        # Store the current working directory path (could be used for relative paths)
        self.path = os.getcwd()
        
        # Store the scaling factor to reduce image resolution and camera parameters
        self.factor = scale_factor
        
        # Load and store all valid image file paths from the provided directory
        self.img_list = self._load_image_paths(img_dir)
        
        # Load camera intrinsic matrix from a file named 'K.txt' in the image directory
        self.K = self._load_camera_matrix(os.path.join(img_dir, 'K.txt'))
        
        # Scale the intrinsic matrix parameters according to the scale factor
        self._scale_camera_matrix()

    def _load_camera_matrix(self, filepath: str) -> np.ndarray:
        """
        Reads the camera intrinsic matrix from a text file.
        
        Parameters:
        - filepath: Path to the text file containing the camera matrix.
        
        Returns:
        - A numpy array of shape (3,3) representing the intrinsic camera matrix.
        """
        with open(filepath, 'r') as file:
            # Read all lines, strip whitespace, and split by lines
            lines = file.read().strip().splitlines()
            
            # Parse each line to convert string values to floats, ignoring empty lines
            matrix = [list(map(float, line.split())) for line in lines if line.strip()]
        
        # Convert the list of lists into a numpy array for matrix operations
        return np.array(matrix)

    def _load_image_paths(self, directory: str) -> List[str]:
        """
        Retrieves the file paths of all images with valid extensions in the given directory.
        
        Parameters:
        - directory: Path to the directory containing images.
        
        Returns:
        - List of full file paths to the images sorted alphabetically.
        """
        # Define allowed image file extensions (case insensitive)
        valid_extensions = {'.jpg', '.png'}
        
        # Filter and sort files in directory that match valid extensions
        image_files = [f for f in sorted(os.listdir(directory)) if os.path.splitext(f)[1].lower() in valid_extensions]
        
        # Construct full file paths for each image file
        return [os.path.join(directory, img) for img in image_files]

    def _scale_camera_matrix(self) -> None:
        """
        Scales the intrinsic camera matrix parameters based on the scaling factor.
        This adjusts the focal lengths and principal point coordinates accordingly.
        """
        # Adjust focal length in x direction
        self.K[0, 0] /= self.factor
        
        # Adjust focal length in y direction
        self.K[1, 1] /= self.factor
        
        # Adjust principal point x-coordinate
        self.K[0, 2] /= self.factor
        
        # Adjust principal point y-coordinate
        self.K[1, 2] /= self.factor

    def scale_down(self, img: np.ndarray) -> np.ndarray:
        """
        Reduces the resolution of the input image by applying Gaussian pyramid downscaling.
        
        Parameters:
        - img: Input image as a numpy array.
        
        Returns:
        - The downscaled image after applying pyrDown multiple times.
        """
        # Calculate how many times to downscale the image based on scale factor
        levels = max(1, int(self.factor / 2))
        
        # Apply Gaussian pyramid downscaling iteratively to reduce image size
        for _ in range(levels):
            img = cv2.pyrDown(img)
        
        # Return the final downscaled image
        return img
