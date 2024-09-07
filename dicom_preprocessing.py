import cv2
import os
import pydicom
import numpy as np
import gradio as gr
from pydicom.pixel_data_handlers.util import apply_voi_lut

BLANK = np.zeros(shape=[512, 288, 3])
MAX_PIXEL_TRES = 180
AREA_PCT_TRES = 0.04

def set_max_pixel_tres(value):
    global MAX_PIXEL_TRES
    MAX_PIXEL_TRES = value

def extract_roi_otsu(data, gkernel=(5, 5), area_pct_tresh=0.004):
    """
    copied with edit from
    https://github.com/dangnh0611/kaggle_rsna_breast_cancer/blob/bf30f8753c564874e1e7862bafd99b09a0471165/src/roi_det/roi_extract.py#L25

    :param data:
    :param gkernel:
    :param area_pct_tresh:
    :return: Tuple(roi_bbox, bolean if success)
    """
    ori_h, ori_w = data.shape[:2]
    img = data.copy()
    # clip percentile: implant, white lines
    upper = np.percentile(img, 95)
    img[img > upper] = np.min(img)
    # Gaussian filtering to reduce noise (optional)
    if gkernel is not None:
        img = cv2.GaussianBlur(img, gkernel, 0.1)
        img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    _, img_bin = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilation to improve contours connectivity
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    img_bin = cv2.dilate(img_bin, element)
    cnts, _ = cv2.findContours(img_bin.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return [0, 0, ori_w, ori_h], False
    areas = np.array([cv2.contourArea(cnt) for cnt in cnts])
    select_idx = np.argmax(areas)
    cnt = cnts[select_idx]
    area_pct = areas[select_idx] / (img.shape[0] * img.shape[1])
    if area_pct < area_pct_tresh:
        return [0, 0, ori_w, ori_h], False
    x0, y0, w, h = cv2.boundingRect(cnt)
    # min-max for safety only
    # x0, y0, x1, y1
    x1 = min(max(int(x0 + w), 0), ori_w)
    y1 = min(max(int(y0 + h), 0), ori_h)
    x0 = min(max(int(x0), 0), ori_w)
    y0 = min(max(int(y0), 0), ori_h)
    return [x0, y0, x1, y1], True


def roi_cropping_image(data, area_pct_tresh=AREA_PCT_TRES):
    (x0, y0, x1, y1), success = extract_roi_otsu(data, (5, 5), area_pct_tresh)
    if success:
        data = data[y0:y1, x0:x1]
    return data, success


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
    padded_img[top_bottom_padding:top_bottom_padding + height, left_right_padding:left_right_padding + width] = img

    return padded_img


def get_laterality(data: np.ndarray, lat=None):
    if lat is None:
        return lat
    # setelah eksperimen lebih lanjut , lebih baik resize sebelum deteksi laterality
    data = cv2.resize(data.copy(), (144, 256), interpolation=cv2.INTER_LINEAR)
    mean_L = np.mean(data[:, :40])
    mean_R = np.mean(data[:, -40:])
    if mean_L == mean_R:
        laterality = "<?>"
    elif (mean_L < MAX_PIXEL_TRES) and (mean_R < MAX_PIXEL_TRES):
        laterality = 'L' if mean_L > mean_R else 'R'
    else:
        laterality = 'L' if mean_L < mean_R else 'R'
    return laterality


def cut_far_pixels(img, laterality, manual_inspected=False):
    if manual_inspected:
        return img
    _, width = img.shape[:2]
    cut = None
    for c in np.arange(0.5, 0.01, -0.01):
        far_cuts = int(c * width)
        if laterality == "L":
            far = img[:, -far_cuts:]
        elif laterality == "R":
            far = img[:, :far_cuts]
        else:
            break
        percentile0 = np.percentile(far, 70)
        percentile1 = np.percentile(far, 95)
        is_save_to_cut = percentile0 > MAX_PIXEL_TRES or percentile1 < 10
        if is_save_to_cut:
            cut = width - far_cuts
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
    return data

def preprocess_image_single(
        img_array: np.ndarray,
        pad_scale: tuple = (1, 1),
        pad_value: int = 0,
        roi_crop: str = None,
        resize: tuple | list = (288, 512),
        return_misch: bool = False,
        lat: str=None,
):
    """

    :param img_array:
    :param pad_scale:
    :param pad_value:
    :param roi_crop:
    :param resize:
    :param return_misch:
    :param lat:
    :return:
    """
    lat_before = get_laterality(img_array, lat=lat)
    # otsu roi cropping
    if roi_crop == 'otsu':
        img_array = cut_far_pixels(img_array, lat_before)
        img_array, _ = roi_cropping_image(img_array)
    elif roi_crop is not None:
        img_array = cut_far_pixels(img_array, lat_before)
    # padding
    if pad_scale is not None:
        img_array = pad_to_scale_ratio(img_array, get_laterality(img_array, lat=lat), pad_scale, pad_value)
    # resize
    if resize is not None:
        img_array = cv2.resize(img_array, resize, interpolation=cv2.INTER_LINEAR)
    pct75 = np.percentile(img_array, 75)
    lat_after = get_laterality(img_array)
    if return_misch:
        return img_array, pct75, lat_before, lat_after
    else:
        return img_array

def read_preprocess_single(path, voi_lut=False, fix_monochrome=False, **kwargs):
    """
    :param path: dicom image path or image bytes io
    :param voi_lut: boolean
    :param fix_monochrome: boolean
    :param return_misc: boolean
    :param pad_scale: padding scale ratio (width, height), default (1, 1)
    :param pad_value: padding value (default 0)
    :param roi_crop: str, "otsu" to apply otsu thresholding + find contours cropping
    :param resize: resize image
    :return:
    """
    data = read_dicom(path, voi_lut=voi_lut, fix_monochrome=fix_monochrome)
    return preprocess_image_single(data, **kwargs)

def preprocess_images_multi_view(
        img_cc: np.ndarray, img_mlo: np.ndarray,
        pad_scale: tuple = (1, 1),
        pad_value: int = 0,
        roi_crop: str = None,
        resize: tuple | list = (288, 512),
        return_misch: bool = False,
        lat: str = None,
        manual_inspected: bool = False,
):
    if manual_inspected:
        assert (lat is not None), "should manual inspect laterality too"
        _, width_cc = img_cc.shape[:2]
        _, width_mlo = img_mlo.shape[:2]
        cut_cc = int(0.5 * width_cc)
        cut_mlo = int(0.5 * width_mlo)
        if lat == "R":
            img_cc = img_cc[:, -cut_cc:]
            img_mlo = img_mlo[:, -cut_mlo:]
        elif lat == "L":
            img_cc = img_cc[:, :cut_cc]
            img_mlo = img_mlo[:, :cut_mlo]
    cc_lat_before = get_laterality(img_cc, lat=lat)
    mlo_lat_before = get_laterality(img_mlo, lat=lat)
    # otsu roi cropping
    if roi_crop == 'otsu':
        img_cc_crop = cut_far_pixels(img_cc, cc_lat_before, manual_inspected=manual_inspected)
        img_mlo_crop = cut_far_pixels(img_mlo, mlo_lat_before, manual_inspected=manual_inspected)
        img_cc_crop, cc_success = roi_cropping_image(img_cc_crop)
        img_mlo_crop, mlo_success = roi_cropping_image(img_mlo_crop)
        if cc_success and mlo_success:
            img_cc = img_cc_crop
            img_mlo = img_mlo_crop
    elif roi_crop is not None:
        img_cc = cut_far_pixels(img_cc, cc_lat_before, manual_inspected=manual_inspected)
        img_mlo = cut_far_pixels(img_mlo, mlo_lat_before, manual_inspected=manual_inspected)
    # padding
    if pad_scale is not None:
        img_cc = pad_to_scale_ratio(img_cc, get_laterality(img_cc, lat=lat), pad_scale, pad_value)
        img_mlo = pad_to_scale_ratio(img_mlo, get_laterality(img_mlo, lat=lat), pad_scale, pad_value)
    # resize
    if resize is not None:
        img_cc = cv2.resize(img_cc, resize, interpolation=cv2.INTER_LINEAR)
        img_mlo = cv2.resize(img_mlo, resize, interpolation=cv2.INTER_LINEAR)
    pct75_cc = np.percentile(img_cc, 75)
    pct75_mlo = np.percentile(img_mlo, 75)
    cc_lat_after = get_laterality(img_cc)
    mlo_lat_after = get_laterality(img_mlo)
    if return_misch:
        return (img_cc, img_mlo), (pct75_cc, pct75_mlo), (cc_lat_before, mlo_lat_before), (cc_lat_after, mlo_lat_after)
    else:
        return img_cc, img_mlo


def read_preprocess_multi_view(path_cc, path_mlo, voi_lut=True, fix_monochrome=True, **kwargs):
    """
    :param path_cc: dicom image path or image bytes io for cc view
    :param path_mlo: dicom image path or image bytes io for mlo view
    :param voi_lut: boolean
    :param fix_monochrome: boolean
    :param return_misc: boolean
    :param pad_scale: padding scale ratio (width, height), default (1, 1)
    :param pad_value: padding value (default 0)
    :param roi_crop: str, "otsu" to apply otsu thresholding + find contours cropping
    :param resize: resize image
    :param return_misc:
    :return:
    """
    data_cc = read_dicom(path_cc, voi_lut=voi_lut, fix_monochrome=fix_monochrome)
    data_mlo = read_dicom(path_mlo, voi_lut=voi_lut, fix_monochrome=fix_monochrome)
    return preprocess_images_multi_view(data_cc, data_mlo, **kwargs)


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
            image = read_preprocess_single(
                list_filepath[0],
                voi_lut=self.voi_lut,
                fix_monochrome=self.fix_monochrome,
                pad_scale=pad_scale,
                roi_crop=crop_method,
                resize=self.resize
            )
            return [image, BLANK]
        image1, image2 = read_preprocess_multi_view(
            list_filepath[0], list_filepath[1],
            voi_lut=self.voi_lut,
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
