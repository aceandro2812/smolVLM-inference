import torch
import os
import logging
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import fitz
from PIL import Image
