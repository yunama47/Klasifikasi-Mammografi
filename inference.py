import numpy as np
import mmv_model
import pathlib
import keras
import glob
import time
import os
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
            pattern = os.path.join(model_path, "*_fold-{}_*.h5")
            self.k = len(glob.glob(pattern.format("*")))
            for fold in range(self.k):
                try:
                    fold_path = glob.glob(pattern.format(fold))[0]
                    model_name = os.path.basename(fold_path)
                    self.print("loading", model_name)
                    self.model[model_name] = keras.models.load_model(fold_path)
                    self.model_list.append(model_name)
                except IndexError as e:
                    self.print(f"model fold {fold} not found")
        else:
            raise UserError(f"Model path should be a HDF5 file or directory. But got {str(model_path)}")

    def inference_single(self, Examined_view: np.ndarray, Auxiliary_view: np.ndarray = None, fold: str = None):
        """
        inference model multi view
        :param fold: number of fold {0,1,2,...,N} or "k-fold ensemble" to use every fold
        :param Examined_view: 3D Image Array (Height:512, Width:288, Channel:3)
        :param Auxiliary_view: 3D Image Array (Height:512, Width:288, Channel:3)
        :return: prediction result array
        """
        if Auxiliary_view is None:
            Auxiliary_view = np.zeros(IMAGE_SIZE)
        assert len(Examined_view.shape) == len(Auxiliary_view.shape) == 3, "images must be 3D Array"
        assert Examined_view.shape == Auxiliary_view.shape == IMAGE_SIZE, "images size must be (Height:512, Width:288, Channel:3)"
        inputs_dict = {
            "Examined": np.expand_dims(Examined_view, 0),
            "Aux": np.expand_dims(Auxiliary_view, 0),
        }
        if fold == "k-fold ensemble" and not isinstance(self.model, keras.Model):
            self.print("prediction of k-fold ensemble models")
            probs = []
            for i in range(self.k):
                pred = self.model[self.model_list[i+1]].predict(inputs_dict, verbose=0)
                probs.append(pred)
            prediction = np.array(probs)
            prediction = np.mean(prediction, axis=0)[0]
        elif isinstance(self.model, keras.Model):
            self.print("prediction of", self.model_list[0])
            prediction = self.model.predict(inputs_dict, verbose=0)[0]
        elif fold in list(self.model_list):
            self.print("prediction of", fold)
            prediction = self.model[fold].predict(inputs_dict, verbose=0)[0]
        else:
            raise UserError("unknown fold.")
        return prediction

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
    infer = ModelInference(models_path, verbose=True, warmup=True)
    start_infer = time.time()
    infer.test_inference()
    finish_infer = time.time()
    print("total time :", finish_infer - start, 'seconds')
    print("inference time :", finish_infer - start_infer, 'seconds')
    print(p, p.dtype)
