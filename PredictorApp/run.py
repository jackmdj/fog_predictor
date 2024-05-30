from app.model import ImageModel, MLPModel
from app.utils import getData

def main():
    text, image = getData()

    Image_Model = ImageModel('saved_models/image_model_scripted.pt')
    MLP_Model = MLPModel('saved_models/mlp_model_scripted.pt')

    img_pred = Image_Model.predict(image)
    mlp_pred = MLP_Model.predict_mlp(text)

    print(f'Image Classifier Prediction: {img_pred}')
    print(f'MLP Prediction: {mlp_pred}')

if __name__ == '__main__':
    main()
