# extract_images_from_nd2_file

Extracts images from ND2 files and outputs in common image formats

Takes ND2 file and outputs the image with user-defined histogram and z project adjustment.

To run:

Adjust settings using the config.yml file
Then run the nd2_to_tif.py script
Requirements:

python 3.9.12
nd2 0.7.1
numpy 1.21.5
opencv-python 4.5.5.64
scikit-image 0.19.2
PyYAML 6.0
Caveats:

Currently doesn't support more than 3 channels for outputing composites
Change log:
