import glob
import os
import pathlib
import gradio as gr
import numpy as np
from dicom_preprocessing import (BLANK,
                                 PreprocessingDICOM,
                                 AdjustImage)
from inference import ModelInference
from downloads import models_download, download_example_dicom

D = PreprocessingDICOM()
A = AdjustImage()

print(" Preparing to launch ".center(50, "="))
print(f"downloading models...")
MODEL_DIR = pathlib.Path(models_download())
print(f"loading models from {MODEL_DIR.absolute()}")
infer = ModelInference(MODEL_DIR)
print(f"downloading example dicom files")
download_example_dicom()
example_dicom_dir = "example_dicom"
print(" Preparation Finished ".center(50, "="))

def dicom_preprocessing_options(option:str):
    assert option in list(vars(D).keys()), gr.Error("invalid option")

    def preprocess(file_list, option_value):
        vars(D)[option] = option_value
        images = D.process_dicom_files(file_list)
        return images

    return preprocess

def images_adjustment_options(option:str):
    assert option in list(vars(A).keys()), gr.Error("invalid option")

    def adjust(img1, img2, option_value):
        vars(A)[option] = option_value
        img1 = A.adjust_contrast_brightness(img1)
        img2 = A.adjust_contrast_brightness(img2)
        return [img1, img2]

    return adjust

def readable_prediction(im1, im2, model_fold):
    pred = infer.inference_single(im1, im2, fold=model_fold)
    actions_pred = [0, 0, 0]
    actions_pred[0]=sum(pred[0:2])
    actions_pred[1]=pred[2]
    actions_pred[2]=sum(pred[3:5])
    result = np.argmax(pred, axis=-1)
    action_result = np.argmax(actions_pred, axis=-1)
    result = [result + 1, pred[result]*100]
    actions = ['"no follow-up"', '"follow-up"', '"biopsy"']
    action_result = [actions[action_result], actions_pred[action_result]*100]
    # detailed probability distribution
    pred = [[f"BI-RADS {i+1}", pred[i]] for i in range(len(pred))]
    pred = sorted(pred, key=lambda x: x[0], reverse=False)
    birads_interpretation = [
        # source : https://radiopaedia.org/articles/breast-imaging-reporting-and-data-system-bi-rads
        "incomplete, need additional imaging evaluation", # BI-RADS 0
        "negative, no lesion found in image", # BI-RADS 1
        "benign, 0% probability of malignancy", # BI-RADS 2
        "probably benign, <2% probability of malignancy", # BI-RADS 3
        "suspicious for malignancy, 2-95% probability of malignancy", # BI-RADS 4
        "highly suggestive of malignancy, >95% probability of malignancy", # BI-RADS 5
        "known biopsy-proven malignancy", # BI-RADS 6
    ]
    prediction_texts = f'''## Predictions : 
    ## - Category BI-RADS : {result[0]} ({result[1]:.3f}%)
    ## - Recommended action : {action_result[0]} ({action_result[1]:.3f}%) 
    > Prediction result is that our model have `{result[1]:.3f}%` confidence that the mammography case is `BI-RADS {result[0]}`. Which mean the case is {birads_interpretation[result[0]]}. Meanwhile our model predict that the recommended action is {action_result[0]}.
    
    All BI-RADS probability prediction:
    '''
    for birads, prob in pred:
        color = 'orange'
        prob *= 100
        if abs(result[1]-prob) <= 1:
            color = 'lime'
        elif abs(result[1]-prob) <= 20:
            color = 'cyan'
        prediction_texts += f'\n * <span style="color: {color};">{birads} : {prob:.3f}% </span>'
    return prediction_texts

def example_fn(*args):
    files = [os.path.join(example_dicom_dir, f) for f in args]
    return files


with gr.Blocks() as demo:
    with gr.Tab("temp", visible=False):
        tmp_image1 = gr.Image(value=BLANK, format="PNG", visible=False)
        tmp_image2 = gr.Image(value=BLANK, format="PNG", visible=False)
        tmp_texbox1 = gr.Textbox()
        tmp_texbox2 = gr.Textbox()
    with gr.Tab("model demo"):
        gr.Markdown(
            """
            # Breast Mammography Classification with Ipsilateral Multi-view model
            ---
            """
        )
        with gr.Row():
            with gr.Column(variant='compact'):
                image1 = gr.Image(value=BLANK, label="ipsilateral view 1", format="PNG", interactive=True, sources=[])
                image2 = gr.Image(value=BLANK, label="ipsilateral view 2", format="PNG", interactive=True, sources=[])
            with gr.Column():
                files_input = gr.Files(label="Input DICOM files (2 Ipsilateral view images)",
                                       file_types=[".dicom", ".DICOM", '.dcm'],
                                       type='filepath')
                examples = [[os.path.basename(path) for path in glob.glob(f"{example_dicom_dir}/*_{birads}_*.dicom")[:2]] for
                            birads in range(1, 6)]
                gr.Examples(examples,
                            inputs=[tmp_texbox1, tmp_texbox2],
                            outputs=files_input,
                            label="Example input DICOM files",
                            fn=example_fn,
                            run_on_click=True)
                with gr.Accordion("Model Predictions", open=True):
                    model_list = infer.get_model_list
                    gr.Markdown(
                        """
                        > Our model will predict the mammography case into two labels : `BI-RADS` and `recommended action`. Where `BI-RADS` is the primary labels which the model trained on, and `recommended action` is a secondary label calculated from `BI-RADS` prediction.
                        """
                    )
                    model_choice = gr.Dropdown(model_list, label="change model", value=model_list[0])
                    predict_btn = gr.Button("Predict", variant="primary")
                    files_input.change(D.process_dicom_files, inputs=files_input, outputs=[tmp_image1, tmp_image2])
                    tmp_image1.change(A.adjust_contrast_brightness, inputs=tmp_image1, outputs=image1)
                    tmp_image2.change(A.adjust_contrast_brightness, inputs=tmp_image2, outputs=image2)

                    default_text = 'Please click the "Predict" button to get BI-RADS prediction'
                    prediction_result = gr.Markdown(default_text)
                    predict_btn.click(readable_prediction, inputs=[image1, image2, model_choice], outputs=prediction_result)
                    files_input.change(lambda: default_text, outputs=prediction_result)

                with gr.Accordion("DICOM Preprocessing Options", open=True):
                    gr.Markdown("""Our model have (H:512, W:288 , C:3) input size, so images will always be resized to `288x512` (Width x Height),i.e, `9:16` ratio
                                * `Apply VOI LUT` : applying VOI LUT function from DICOM
                                * `Fix MONOCHROME1 interpretation` : do negative inversion if DICOM images are in MONOCHROME1 interpretation
                                * `Padding to 9:16 aspect-ratio` : padding so that the image is not distorted when resized
                                * `Breast ROI Crop` : crop breast region only from the images, reducing unused pixels""")
                    apply_voi_lut = gr.Checkbox(label='Apply VOI LUT', value=D.voi_lut)
                    apply_voi_lut.change(dicom_preprocessing_options('voi_lut'),
                                         inputs=[files_input, apply_voi_lut], outputs=[tmp_image1, tmp_image2])
                    fix_monochrome = gr.Checkbox(label='Fix MONOCHROME1 interpretation', value=D.fix_monochrome)
                    fix_monochrome.change(dicom_preprocessing_options('fix_monochrome'),
                                          inputs=[files_input, fix_monochrome], outputs=[tmp_image1, tmp_image2])
                    padding = gr.Checkbox(label='Padding to 9:16 aspect-ratio', value=D.padding)
                    padding.change(dicom_preprocessing_options('padding'),
                                   inputs=[files_input, padding], outputs=[tmp_image1, tmp_image2])
                    roi_crop = gr.Checkbox(label="Breast ROI Crop", value=D.roi_crop)
                    roi_crop.change(dicom_preprocessing_options('roi_crop'),
                                    inputs=[files_input, roi_crop], outputs=[tmp_image1, tmp_image2])

                with gr.Accordion("Image Adjustment Options", open=False):
                    gr.Markdown("**INFO** : This will apply on both images")
                    slider_ct = gr.Slider(minimum=0, maximum=3,
                                          value=A.contrast_factor, label="adjust contrast")
                    slider_ct.change(images_adjustment_options("contrast_factor"),
                                     inputs=[tmp_image1, tmp_image2, slider_ct], outputs=[image1, image2])
                    slider_br = gr.Slider(minimum=-200, maximum=200,
                                          value=A.brightness_factor, label="adjust brightness")
                    slider_br.change(images_adjustment_options("brightness_factor"),
                                     inputs=[tmp_image1, tmp_image2, slider_br], outputs=[image1, image2])
                    reset = gr.Button("reset adjustment", size='sm')
                    reset.click(lambda: (1, 0), outputs=[slider_ct, slider_br])


if __name__ == "__main__":
    demo.launch()
