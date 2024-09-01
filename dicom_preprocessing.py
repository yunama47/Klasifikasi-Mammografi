import cv2
import os
import pydicom
import numpy as np
import gradio as gr
from pydicom.pixel_data_handlers.util import apply_voi_lut

BLANK = np.zeros(shape=[512, 288, 3])

def extract_roi_otsu(data, gkernel=(5, 5), area_pct_tresh=0.004):
    ori_h, ori_w = data.shape[:2]
    img = data.copy()
    # Gaussian filtering to reduce noise (optional)
    if gkernel is not None:
        img = cv2.GaussianBlur(img, gkernel, 0)
    _, img_bin = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilation to improve contours connectivity
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    img_bin = cv2.dilate(img_bin, element)
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return [0, 0, ori_w, ori_h]
    areas = np.array([cv2.contourArea(cnt) for cnt in cnts])
    select_idx = np.argmax(areas)
    cnt = cnts[select_idx]
    area_pct = areas[select_idx] / (img.shape[0] * img.shape[1])
    if area_pct < area_pct_tresh:
        return [0, 0, ori_w, ori_h]
    x0, y0, w, h = cv2.boundingRect(cnt)
    # min-max for safety only
    # x0, y0, x1, y1
    x1 = min(max(int(x0 + w), 0), ori_w)
    y1 = min(max(int(y0 + h), 0), ori_h)
    x0 = min(max(int(x0), 0), ori_w)
    y0 = min(max(int(y0), 0), ori_h)
    return [x0, y0, x1, y1]

def otsu_roi_crop(data, area_pct_tresh=0.03):
    x0, y0, x1, y1 = extract_roi_otsu(data, (3, 3), area_pct_tresh)
    data = data[y0:y1,x0:x1]
    return data

def pad_to_scale_ratio(img, laterality, scale, pad_value=0):
    height, width = img.shape[:2]
    target_ratio = scale[0] / scale[1]

    if width / height > target_ratio:  # Image is wider than 9:16
        new_width = width
        new_height = int(width / target_ratio)
        top_bottom_padding = (new_height - height) // 2
        left_right_padding = 0
    else:  # Image is taller than 9:16
        new_width = int(height * target_ratio)
        new_height = height
        top_bottom_padding = 0
        if laterality == "R":
            left_right_padding = (new_width - width)
        else:
            left_right_padding = 0

    # Create a blank image with the target dimensions
    padded_img = np.ones((new_height, new_width), dtype=np.uint8) * pad_value

    # Copy the original image onto the padded image with the calculated offsets
    padded_img[top_bottom_padding:top_bottom_padding + height,
               left_right_padding:left_right_padding + width] = img

    return padded_img

def get_laterality(data: np.ndarray, laterality=None):
    if laterality is None:
        sum_L = np.sum(data[:, :20])
        sum_R = np.sum(data[:, -20:])
        if sum_L == sum_R:
            laterality = "<?>"
        else:
            laterality = 'L' if sum_L > sum_R else 'R'
    return laterality


def cut_far_pixels(img, laterality):
    _, width = img.shape[:2]
    cut = None
    for c in np.arange(0.01, 0.41, 0.02):
        far_cuts = int(c*width)
        if laterality == "L":
            far = img[:, -far_cuts:]
        elif laterality == "R":
            far = img[:, :far_cuts]
        else:
            break
        percentile = np.percentile(far, 80)
        is_save_to_cut = 10 > percentile or 160 < percentile
        if is_save_to_cut:
            cut = width-far_cuts
        else:
            break
    if laterality == "L":
        return img[:, :cut]
    elif laterality == "R":
        return img[:, -(cut or 0):]
    else:
        return img

def read_dicom(path,
               voi_lut: bool = True,
               fix_monochrome: bool = True,
               pad_scale: tuple = (1, 1),
               pad_value: int = 0,
               roi_crop: str = None,
               resize: tuple|list = (288, 512),
               return_misch: bool=False,
               laterality: str = None,
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
    # otsu roi cropping
    if roi_crop == 'otsu':
        data = cut_far_pixels(data, get_laterality(data, laterality))
        data = otsu_roi_crop(data)
    elif roi_crop is not None:
        data = cut_far_pixels(data, get_laterality(data, laterality))
    median = np.median(data)
    # padding
    if pad_scale is not None:
        data = pad_to_scale_ratio(data, get_laterality(data, laterality), pad_scale, pad_value)
    # resize
    if resize is not None:
        data = cv2.resize(data, resize, interpolation=cv2.INTER_LINEAR)
    if return_misch:
        return data, median
    else:
        return data

class PreprocessingDICOM:
    def __init__(self):
        self.voi_lut = True
        self.fix_monochrome = True
        self.padding = True
        self.resize = (288, 512)
        self.roi_crop = True

    def process_dicom_files(self, list_filepath: list):
        if list_filepath is None:
            return [BLANK, BLANK]
        pad_scale = (9, 16) if self.padding else None
        crop_method = "otsu" if self.roi_crop else None
        if len(list_filepath) > 2:
            dropped = "\n".join([os.path.basename(f) for f in list_filepath[2:]])
            list_filepath = list_filepath[:2]
            gr.Warning(f"Using only first two files, this files is not used: \n{dropped}")
        elif len(list_filepath) < 2:
            image = read_dicom(
                path=list_filepath[0],
                fix_monochrome=self.fix_monochrome,
                pad_scale=pad_scale,
                roi_crop=crop_method,
                resize=self.resize
            )
            return [image, BLANK]
        image1 = read_dicom(
            path=list_filepath[0],
            fix_monochrome=self.fix_monochrome,
            pad_scale=pad_scale,
            roi_crop=crop_method,
            resize=self.resize
        )
        image2 = read_dicom(
            path=list_filepath[1],
            fix_monochrome=self.fix_monochrome,
            pad_scale=pad_scale,
            roi_crop=crop_method,
            resize=self.resize
        )
        if get_laterality(image1) != get_laterality(image2):
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