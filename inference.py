import numpy as np
import mmv_model
import keras
import glob
import os
IMAGE_SIZE = (512, 288, 3)

class UserError(Exception):
    pass

class ModelInference:
    def __init__(self, model_path: str):
        """
        model inference class
        :param model_path: keras hdf5 model path or directory containing keras hdf5 files
        """
        if model_path.endswith('.h5'):
            self.models: keras.Model = keras.models.load_model(model_path)
        elif os.path.isdir(model_path):
            self.models: dict = {}
            pattern = os.path.join(model_path, "*_fold-{}_*.h5")
            self.total_files = len(glob.glob(pattern.format("*")))
            for fold in range(self.total_files):
                try:
                    fold_path = glob.glob(pattern.format(fold))[0]
                    print("loading", fold_path)
                    self.models[f"fold-{fold}"] = keras.models.load_model(fold_path)
                except IndexError as e:
                    print(f"model fold {fold} not found")
        else:
            raise UserError("Model path should be a HDF5 file or directory.")

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
        if isinstance(self.models, keras.Model):
            print("prediction of single fold")
            prediction = self.model.predict(inputs_dict, verbose=0)[0]
        elif fold is not None and fold in list(self.models.keys()):
            print("prediction of fold", fold)
            prediction = self.models[fold].predict(inputs_dict, verbose=0)[0]
        elif fold == "k-fold ensemble":
            print("prediction of k-fold ensemble")
            probs = []
            for fold in range(self.total_files):
                pred = self.models[f"fold-{fold}"].predict(inputs_dict, verbose=0)
                probs.append(pred)
            prediction = np.array(probs)
            prediction = np.mean(prediction, axis=0)[0]
        else:
            raise UserError("unknown fold.")
        return prediction


if __name__ == '__main__':
    import time
    start = time.time()
    infer = ModelInference("model")
    img1 = np.random.uniform(0, 255, size=IMAGE_SIZE)
    img2 = np.random.uniform(0, 255, size=IMAGE_SIZE)
    p = infer.inference_single(img1, img2, fold="k-fold ensemble")
    print("total time :", time.time() - start, 'seconds')
