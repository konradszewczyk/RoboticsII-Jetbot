import cv2
import onnxruntime as rt

from jetbot import Camera
from pathlib import Path
import yaml
import numpy as np

import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PUTDriver import PUTDriver


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        ##TODO: preprocess your input image, remember that img is in BGR channels order
        #raise NotImplementedError
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = img / 255
        img = torch.from_numpy(img)
        img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = np.expand_dims(img.numpy(axis=0))
        return img

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        ##TODO: prepare your outputs
        #raise NotImplementedError
        detections = np.clip(detections, -0.9999, 0.9999)
        return detections[0]

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() < 1.0
        assert outputs.min() > -1.0

        return outputs


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    camera = Camera.instance(width=224, height=224)

    # model warm-up
    image = camera.value
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        image = camera.value
        forward, left = ai.predict(image)


if __name__ == '__main__':
    main()
