import glob
import os
import pathlib
import gradio as gr
import numpy as np
import pandas as pd
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

def readable_prediction(im1, im2, model_fold, lang):
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
    if lang=='en':
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
        prediction_texts = f'''<h2>Prediction :</h2>
        <h2>BI-RADS Category : {result[0]} ({result[1]:.3f}%)</h2>
        <h2>Recommended Action : {action_result[0]} ({action_result[1]:.3f}%)</h2>
        <p>The prediction result is that our model has a confidence of <strong>{result[1]:.3f}%</strong> that this mammography case belongs to <strong>BI-RADS {result[0]}</strong>. This means the case is {birads_interpretation[result[0]]}. Meanwhile, our model predicts that the recommended action is {action_result[0]}.</p>
        <p>All BI-RADS probability predictions:</p>
        '''
    elif lang=='id':
        birads_interpretation = [
            # sumber : https://radiopaedia.org/articles/breast-imaging-reporting-and-data-system-bi-rads
            "tidak lengkap, perlu evaluasi pencitraan tambahan", # BI-RADS 0
            "negatif, tidak ada lesi yang ditemukan dalam gambar", # BI-RADS 1
            "jinak, 0% kemungkinan keganasan", # BI-RADS 2
            "mungkin jinak, <2% kemungkinan keganasan", # BI-RADS 3
            "mencurigakan ganas, 2-95% kemungkinan keganasan", # BI-RADS 4
            "sangat menunjukkan keganasan, >95% kemungkinan keganasan", # BI-RADS 5
            "keganasan yang diketahui dan terbukti dengan biopsi", # BI-RADS 6
        ]
        prediction_texts = f'''<h2>Prediksi :</h2>
        <h2>Kategori BI-RADS : {result[0]} ({result[1]:.3f}%)</h2>
        <h2>Rekomendasi Aksi : {action_result[0]} ({action_result[1]:.3f}%)</h2>
        <p>Hasil prediksi adalah bahwa model kami memiliki kepercayaan <strong>{result[1]:.3f}%</strong> bahwa kasus mamografi ini tergolong <strong>BI-RADS {result[0]}</strong>. Yang berarti kasus tersebut adalah {birads_interpretation[result[0]]}. Sementara itu, model kami memprediksi bahwa tindakan yang direkomendasikan adalah {action_result[0]}.</p>
        <p>Semua prediksi probabilitas BI-RADS:</p>
        '''
    else:
        raise ValueError(f"only 'en' and 'id' are supported, got {lang}")
    
    for birads, prob in pred:
        color = 'orange'
        prob *= 100
        if abs(result[1]-prob) <= 1:
            color = 'lime'
        elif abs(result[1]-prob) <= 20:
            color = 'cyan'
        prediction_texts += f'\n - <span style="color: {color};">{birads} : {prob:.3f}% </span><br>'
    return prediction_texts

def example_fn(*args):
    files = [os.path.join(example_dicom_dir, f) for f in args]
    return files

def bilingual_content(lang):
    translation = pd.read_csv("translation.csv").set_index(["id", "lang"])
    gr.HTML(
            f"""
            <div style="text-align: center;">
                <h1>{translation.loc[("title", lang.value), "value"]}</h1>
                <hr>
                <a href="https://www.kaggle.com/models/gedewahyupurnama/mammographymultiview/" target="_blank">
                    <div style="display: inline-flex; align-items: center; gap: 2px">
                        <p>{translation.loc[("subtitle", lang.value), "value"]}</p>
                        <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png" width="50px">
                    </div>
                </a>
            </div>
            """
        )
    with gr.Row():
        with gr.Column(variant='compact'):
            image1 = gr.Image(value=BLANK, label="ipsilateral view 1", format="PNG", interactive=True, sources=[])
            image2 = gr.Image(value=BLANK, label="ipsilateral view 2", format="PNG", interactive=True, sources=[])
        with gr.Column():
            files_input = gr.Files(label=translation.loc[("file_upload_label", lang.value), "value"],
                                    file_types=[".dicom", ".DICOM", '.dcm'],
                                    type='filepath')
            examples = [[os.path.basename(path) for path in glob.glob(f"{example_dicom_dir}/*_{birads}_*.dicom")[:2]] for
                        birads in range(1, 6)]
            gr.Examples(examples,
                        inputs=[tmp_texbox1, tmp_texbox2],
                        outputs=files_input,
                        label=translation.loc[("example_label", lang.value), "value"],
                        fn=example_fn,
                        run_on_click=True)
            with gr.Accordion(translation.loc[("accordion1_label", lang.value), "value"], open=True):
                model_list = infer.get_model_list
                gr.Markdown(
                    f"""
                    > {translation.loc[("accordion1_desc", lang.value), "value"]}
                    """
                )
                model_choice = gr.Dropdown(model_list, label="change model", value=model_list[0])
                predict_btn = gr.Button("Predict", variant="primary")
                files_input.change(D.process_dicom_files, inputs=files_input, outputs=[tmp_image1, tmp_image2])
                tmp_image1.change(A.adjust_contrast_brightness, inputs=tmp_image1, outputs=image1)
                tmp_image2.change(A.adjust_contrast_brightness, inputs=tmp_image2, outputs=image2)

                default_text = f"<p>{translation.loc[('predict_result_devault', lang.value), 'value']}</p>"
                prediction_result = gr.HTML(default_text)
                predict_btn.click(readable_prediction, inputs=[image1, image2, model_choice, lang], outputs=prediction_result)
                files_input.change(lambda: default_text, outputs=prediction_result)

            with gr.Accordion(translation.loc[("accordion2_label", lang.value), "value"], open=True):
                gr.Markdown(translation.loc[("accordion2_desc", lang.value), "value"])
                apply_voi_lut = gr.Checkbox(label=translation.loc[("apply_voi_lut", lang.value), "value"], value=D.voi_lut)
                apply_voi_lut.change(dicom_preprocessing_options('voi_lut'),
                                        inputs=[files_input, apply_voi_lut], outputs=[tmp_image1, tmp_image2])
                fix_monochrome = gr.Checkbox(label=translation.loc[("fix_monochrome", lang.value), "value"], value=D.fix_monochrome)
                fix_monochrome.change(dicom_preprocessing_options('fix_monochrome'),
                                        inputs=[files_input, fix_monochrome], outputs=[tmp_image1, tmp_image2])
                padding = gr.Checkbox(label=translation.loc[("padding_aspect_ratio", lang.value), "value"], value=D.padding)
                padding.change(dicom_preprocessing_options('padding'),
                                inputs=[files_input, padding], outputs=[tmp_image1, tmp_image2])
                roi_crop = gr.Checkbox(label=translation.loc[("breast_roi_crop", lang.value), "value"], value=D.roi_crop)
                roi_crop.change(dicom_preprocessing_options('roi_crop'),
                                inputs=[files_input, roi_crop], outputs=[tmp_image1, tmp_image2])

            with gr.Accordion(translation.loc[("accordion3_label", lang.value), "value"], open=False):
                gr.Markdown(translation.loc[("accordion3_desc", lang.value), "value"])
                slider_ct = gr.Slider(minimum=0, maximum=3,
                                        value=A.contrast_factor, label=translation.loc[("contrast", lang.value), "value"])
                slider_ct.change(images_adjustment_options("contrast_factor"),
                                    inputs=[tmp_image1, tmp_image2, slider_ct], outputs=[image1, image2])
                slider_br = gr.Slider(minimum=-200, maximum=200,
                                        value=A.brightness_factor, label=translation.loc[("brightness", lang.value), "value"])
                slider_br.change(images_adjustment_options("brightness_factor"),
                                    inputs=[tmp_image1, tmp_image2, slider_br], outputs=[image1, image2])
                reset = gr.Button(translation.loc[("reset_adjustment", lang.value), "value"], size='sm')
                reset.click(lambda: (1, 0), outputs=[slider_ct, slider_br])

with gr.Blocks() as demo:
    with gr.Tab("temp", visible=False):
        tmp_image1 = gr.Image(value=BLANK, format="PNG", visible=False)
        tmp_image2 = gr.Image(value=BLANK, format="PNG", visible=False)
        tmp_texbox1 = gr.Textbox()
        tmp_texbox2 = gr.Textbox()

    with gr.Tab("english".capitalize()):
        lang = gr.Textbox("en",visible=False)
        bilingual_content(lang)

    with gr.Tab("bahasa indonesia".capitalize()):
        lang = gr.Textbox("id",visible=False)
        bilingual_content(lang)

if __name__ == "__main__":
    demo.launch()
