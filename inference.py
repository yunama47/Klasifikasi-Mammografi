import tensorflow as tf
import numpy as np
import mmv_model
import pathlib
import keras
import glob
import time
import os

from typing import Iterable, Sized
IMAGE_SIZE = (512, 288, 3)
class UserError(Exception):
    pass

class ModelInference:
    def __init__(self, model_path: pathlib.Path, verbose=False):
        """
        model inference class
        :param model_path: keras hdf5 model path or directory containing keras hdf5 files
        """
        if verbose:
            self.print = print
        else:
            self.print = lambda *args: None
        if model_path.is_file() and model_path.__str__().endswith('.h5'):
            self.model: keras.Model = keras.models.load_model(model_path)
            self.model_list = [os.path.basename(model_path)]
        elif model_path.is_dir():
            self.model: dict = {}
            self.model_list = ["k-fold ensemble"]
            pattern = os.path.join(model_path, "*.h5")
            self.k = len(glob.glob(pattern))
            for fold_path in glob.glob(pattern):
                model_name = os.path.basename(fold_path)
                self.print("loading", model_name)
                self.model[model_name] = keras.models.load_model(fold_path)
                self.model_list.append(model_name)
        else:
            raise UserError(f"Model path should be a Tensorflow's .h5 model file or directory. But got {str(model_path)}")

    def __get_prediction(self, inputs, fold, output_size):
        if fold == "k-fold ensemble" and not isinstance(self.model, keras.Model):
            self.print("prediction of k-fold ensemble models")
            pred_sum = np.zeros(shape=output_size, dtype="float64")
            for i in range(self.k):
                pred = self.model[self.model_list[i + 1]].predict(inputs)
                if isinstance(pred, dict):
                    pred = pred["birads"]
                pred = tf.nn.softmax(pred).numpy()
                pred_sum += pred
            prediction = pred_sum / self.k
            return prediction
        elif isinstance(self.model, keras.Model):
            self.print("prediction of", self.model_list[0])
            prediction = self.model.predict(inputs, verbose=1)
        elif fold in list(self.model_list):
            self.print("prediction of", fold)
            prediction = self.model[fold].predict(inputs, verbose=1)
        else:
            raise UserError("unknown fold.")
        if isinstance(prediction, dict):
            prediction = prediction["birads"]
        return tf.nn.softmax(prediction).numpy()

    def inference_single(self, CC_view: np.ndarray, MLO_view: np.ndarray = None, fold: str = None):
        """
        inference model multi view
        :param fold: number of fold {0,1,2,...,N} or "k-fold ensemble" to use every fold
        :param CC_view: 3D Image Array (Height:512, Width:288, Channel:3)
        :param MLO_view: 3D Image Array (Height:512, Width:288, Channel:3)
        :return: prediction result array
        """
        if MLO_view is None:
            MLO_view = np.zeros(IMAGE_SIZE)
        assert len(CC_view.shape) == len(MLO_view.shape) == 3, "images must be 3D Array"
        assert CC_view.shape == MLO_view.shape == IMAGE_SIZE, "images size must be (Height:512, Width:288, Channel:3)"
        inputs_dict = {
            "CC": np.expand_dims(CC_view, 0),
            "MLO": np.expand_dims(MLO_view, 0),
        }
        prediction = self.__get_prediction(inputs_dict, fold, (1, 5))
        return prediction[0]

    def inference_batch(self, ds: tf.data.Dataset, fold: str = None, batch_size: int = 2):
        """
        batch inference model multi view
        :param ds: tf.data.Dataset containing "CC" and "MLO" images
        :param fold: number of fold {0,1,2,...,N} or "k-fold ensemble" to use every fold
        :param batch_size: batch size for inference
        :return:
        """
        ds, cardinality = self.check_n_prep_ds(ds, batch_size)
        prediction = self.__get_prediction(ds, fold, (cardinality, 5))
        return prediction
    @staticmethod
    def check_n_prep_ds(ds: tf.data.Dataset, batch_size=2):
        unbatchSpec = tf.data.DatasetSpec({
            "CC": tf.TensorSpec(shape=IMAGE_SIZE, dtype=tf.float32),
            "MLO": tf.TensorSpec(shape=IMAGE_SIZE, dtype=tf.float32),
        })
        batchSpec = tf.data.DatasetSpec({
            "CC": tf.TensorSpec(shape=[None, *IMAGE_SIZE], dtype=tf.float32),
            "MLO": tf.TensorSpec(shape=[None, *IMAGE_SIZE], dtype=tf.float32),
        })
        if not isinstance(ds, tf.data.Dataset):
            raise UserError("ds must be a tf.data.Dataset")
        if unbatchSpec.is_compatible_with(ds):
            cardinality = ds.cardinality().numpy()
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            return ds, cardinality
        elif batchSpec.is_compatible_with(ds):
            raise UserError(f"""ds must be compatible with:
            {unbatchSpec}
            please dont batch your dataset
            """)
        else:
            raise UserError(f"""ds must be compatible with:
            {unbatchSpec}
            """)

    def test_inference(self):
        image1 = np.random.randint(0, 255, size=IMAGE_SIZE)
        image2 = np.random.randint(0, 255, size=IMAGE_SIZE)
        _ = self.inference_single(image1, image2, fold="k-fold ensemble")

    @property
    def get_model_list(self):
        if self.model_list is None:
            return []
        elif isinstance(self.model_list, dict):
            return list(self.model_list.values())
        elif isinstance(self.model_list, list):
            return self.model_list


if __name__ == '__main__':
    start = time.time()
    from downloads import models_download
    models_path = pathlib.Path(models_download())
    infer = ModelInference(models_path, verbose=True)
    start_infer = time.time()
    infer.test_inference()
    finish_infer = time.time()
    print("total time :", finish_infer - start, 'seconds')
    print("inference time :", finish_infer - start_infer, 'seconds')
