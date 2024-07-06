import cv2
import os
import pydicom
import numpy as np
import gradio as gr
from pydicom.pixel_data_handlers.util import apply_voi_lut

BLANK = np.zeros(shape=[512, 288, 3])

def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (3, 3), 0.1)
    img_16bit = cv2.normalize(blur, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    _, breast_mask = cv2.threshold(img_16bit,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)

def otsu_roi_crop(data):
    xmin, ymin, w, h = crop_coords(data)
    xmax = xmin + w
    ymax = ymin + h
    data = data[ymin:ymax,xmin:xmax]
    return data

def pad_to_9_16(img):
    height, width = img.shape[:2]
    target_ratio = 9 / 16

    if width / height > target_ratio:  # Image is wider than 9:16
        new_width = width
        new_height = int(width / target_ratio)
        top_bottom_padding = (new_height - height) // 2
        left_right_padding = 0
    else:  # Image is taller than 9:16
        new_width = int(height * target_ratio)
        new_height = height
        top_bottom_padding = 0
        left_right_padding = (new_width - width) // 2

    # Create a blank image with the target dimensions
    padded_img = np.zeros((new_height, new_width), dtype=np.uint8)

    # Copy the original image onto the padded image with the calculated offsets
    padded_img[top_bottom_padding:top_bottom_padding + height,
               left_right_padding:left_right_padding + width] = img

    return padded_img


def read_dicom(path,
               voi_lut:bool=True,
               fix_monochrome:bool=True,
               pad:bool=False,
               roi_crop:str=None,
               resize:tuple|list=(288, 512)
               ):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    sum_L = np.sum(data[:, :2])
    sum_R = np.sum(data[:, -2:])
    laterality = 'L' if sum_L > sum_R else 'R'
    # otsu roi cropping
    if roi_crop == 'otsu':
        data = otsu_roi_crop(data)
    # padding
    if pad:
        data = pad_to_9_16(data)
    # resize
    if resize is not None:
        data = cv2.resize(data, resize, interpolation=cv2.INTER_LINEAR)

    return data, laterality

class PreprocessingDICOM:
    def __init__(self):
        self.voi_lut = True
        self.fix_monochrome = True
        self.padding = True
        self.resize = (288, 512)
        self.roi_crop = "otsu"

    def process_dicom_files(self, list_filepath:list):
        if list_filepath is None:
            return [BLANK, BLANK]
        if len(list_filepath) > 2:
            dropped = "\n".join([os.path.basename(f) for f in list_filepath[2:]])
            list_filepath = list_filepath[:2]
            gr.Warning(f"Using only first two files, this files is not used: \n{dropped}")
        elif len(list_filepath) < 2:
            image, _ = read_dicom(
                path=list_filepath[0],
                fix_monochrome=self.fix_monochrome,
                pad=self.padding,
                roi_crop=self.roi_crop,
                resize=self.resize
            )
            return [image, BLANK]
        image1, laterality1 = read_dicom(
            path=list_filepath[0],
            fix_monochrome=self.fix_monochrome,
            pad=self.padding,
            roi_crop=self.roi_crop,
            resize=self.resize
        )
        image2, laterality2 = read_dicom(
            path=list_filepath[1],
            fix_monochrome=self.fix_monochrome,
            pad=self.padding,
            roi_crop=self.roi_crop,
            resize=self.resize
        )
        if laterality1 != laterality2:
            gr.Warning("Fist two images is bilateral views")
        return [image1, image2]

class AdjustImage:
    def __init__(self):
        self.contrast_factor = 1
        self.brightness_factor = 0
        self.images1 = None
        self.images2 = None

    def set_images(self, image1, image2):
        self.images1 = image1
        self.images2 = image2

    def adjust_images(self):
        img1 = self.adjust_contrast_brightness(self.images1)
        img2 = self.adjust_contrast_brightness(self.images2)
        return [img1, img2]

    def adjust_contrast_brightness(self, image):
        if image is None:
            return BLANK
        if isinstance(image, dict):
            image = image["composite"]
        contrast = self.contrast_factor
        brightness = self.brightness_factor
        img = image.astype(np.float32)
        # adjust contrast
        mean = np.mean(img)
        img = (img - mean) * contrast + mean
        img = np.clip(img, 0, 255)
        # adjust brightness
        img = np.clip(img + brightness, 0, 255)
        return img.astype(np.uint8)

__all__ = ["PreprocessingDICOM", "AdjustImage"]