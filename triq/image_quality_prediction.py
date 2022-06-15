import glob


from triq.models.triq_model import create_triq_model
import numpy as np
from PIL import Image

from scipy.stats import norm
#
# def predict_image_quality(model_weights_path, image_path):
#     image = Image.open(image_path)
def predict_image_quality(model_weights_path):
    model = create_triq_model(n_quality_levels=5)
    model.load_weights(model_weights_path)

    triq_rates = []

    for i, path in enumerate(glob.glob("temp/*")):
        image = Image.open(path)
        image = np.asarray(image, dtype=np.float32)
        image /= 127.5
        image -= 1.

        prediction = model.predict(np.expand_dims(image, axis=0))

        mos_scales = np.array([1, 2, 3, 4, 5])
        predicted_mos = (np.sum(np.multiply(mos_scales, prediction[0])))
        # print('Predicted MOS: {}'.format(predicted_mos))
        pdf = norm.cdf(predicted_mos, loc=3, scale=0.6666666666666666666666666666666)
        if pdf > 3:
            pdf = 1 - pdf
        triq_rates.append((np.float64(pdf), "desc", path))

    return triq_rates


if __name__ == '__main__':
    image_path = r'.\examples\sample_data\640px-Natchez-Trace-Parkway-Highsmith.jpeg'
    # image_path = r'examples/sample_data/example_image_2 (mos=2.865).jpg'
    # model_weights_path = r'C:\Users\snahir\Desktop\uni\3A\סדנה פרוייקטים\transformer\triq\triq\src\pretrained_weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model_weights_path=r'.\pretrained_weights\TRIQ.h5'
    predict_mos = predict_image_quality(model_weights_path, image_path)
    print('Predicted MOS: {}'.format(predict_mos))

def triq_pred():
    model_weights_path = r'triq/src/pretrained_weights/TRIQ.h5'
    predict_mos = predict_image_quality(model_weights_path)
    return predict_mos
