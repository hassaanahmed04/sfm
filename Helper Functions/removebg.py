import os
from rembg import remove
from PIL import Image
import io

# Input and output folders
input_folder = "iron"
output_folder = "iron2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
supported_formats = ('.jpg', '.jpeg')

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(supported_formats):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")  # Save as PNG

        with open(input_path, "rb") as input_file:
            input_data = input_file.read()
            output_data = remove(input_data)

        with open(output_path, "wb") as output_file:
            output_file.write(output_data)

        print(f"Processed: {filename}")

