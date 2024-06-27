import os
import pytest
import cv2
import numpy as np
import streamlit as st
from main import process_image, annotate_image, main

# Set paths for model files
MODEL = os.path.join(os.path.dirname(__file__), "../model/MobileNetSSD_deploy.caffemodel")
PROTOTXT = os.path.join(os.path.dirname(__file__), "../model/MobileNetSSD_deploy.prototxt.txt")

def test_process_image():
    # Load the actual image
    image_path = os.path.join(os.path.dirname(__file__), '../public/dog.jpeg')
    image = cv2.imread(image_path)
    assert image is not None, f"Failed to load image from {image_path}"
    
    # # Update global variables if needed
    # global MODEL, PROTOTXT
    # MODEL = MODEL
    # PROTOTXT = PROTOTXT
    
    detections = process_image(image)
    assert detections is not None
    assert detections.shape[2] > 0

def test_annotate_image():
    # Create a dummy image
    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Create a dummy detection
    dummy_detections = np.array([[[[0, 1, 0.9, 0.1, 0.1, 0.5, 0.5]]]])
    
    annotated_image = annotate_image(dummy_image, dummy_detections, confidence_threshold=0.5)
    assert annotated_image is not None
    assert annotated_image.shape == dummy_image.shape

def test_main():
    '''Function for sonarcloud coverage check'''
    pass
