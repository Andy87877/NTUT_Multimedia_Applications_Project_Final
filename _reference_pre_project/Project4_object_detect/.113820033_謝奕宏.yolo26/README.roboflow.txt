
project4_yolo_v3 - v1 2026-05-02 4:22pm
==============================

This dataset was exported via roboflow.com on May 4, 2026 at 1:14 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 243 images.
Project4-yolo-v3 are annotated in YOLO26 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Fit (white edges))

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 10 percent of the image
* Random rotation of between -5 and +5 degrees
* Random shear of between -5° to +5° horizontally and -5° to +5° vertically
* Random brigthness adjustment of between -20 and +20 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 1 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


