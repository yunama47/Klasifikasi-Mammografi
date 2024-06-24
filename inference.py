import numpy as np
import MMV_Model
import keras

IMAGE_SIZE = (512, 288, 3)

class ModelInference:
    def __init__(self, model_path: str):
        self.model: keras.Model = keras.models.load_model(model_path)

    def reload_model(self, model_path: str):
        self.model: keras.Model = keras.models.load_model(model_path)

    def inference_single(self, Examined_view: np.ndarray, Auxiliary_view: np.ndarray = None):
        """
        inference model multi view
        :param Examined_view: 3D Image Array (Height:512, Width:288, Channel:3)
        :param Auxiliary_view: 3D Image Array (Height:512, Width:288, Channel:3)
        :return: prediction result array
        """
        if Auxiliary_view is None:
            Auxiliary_view = np.zeros(IMAGE_SIZE)
        assert len(Examined_view.shape) == len(Auxiliary_view.shape) == 3, "images must be 3D Array"
        assert Examined_view.shape == Auxiliary_view.shape == IMAGE_SIZE, f"images size must be (Height:512, Width:288, Channel:3) "
        inputs_dict = {
            "Examined": np.expand_dims(Examined_view, 0),
            "Aux": np.expand_dims(Auxiliary_view, 0),
        }
        prediction = self.model.predict(inputs_dict, verbose=0)
        return prediction[0]


if __name__ == '__main__':
    infer = ModelInference("convnext_dual_view.h5")
    img1 = np.random.uniform(0, 255, size=IMAGE_SIZE)
    img2 = np.random.uniform(0, 255, size=IMAGE_SIZE)
    pred = infer.inference_single(img1, img2)
