import torch

class ImageModel:
    def __init__(self, path):
        self.model = torch.jit.load(path)
        self.model.eval()

    def predict(self, image):
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
        
        # Get the predicted class probabilities
        probabilities = torch.softmax(output, dim=1)
        
        # Get the predicted class (0 or 1 for binary classification)
        predicted_class = probabilities.argmax(dim=1).item()
        
        # Get the probability of the predicted class
        probability = probabilities[0, predicted_class].item()
        result = "No Fog" if predicted_class == 1 else "Fog"
        probability_percent = probability * 100  # Convert to percentage
        
        
        return {"prediction accuracy": f"{probability_percent:.4f}%", "prediction": result}
    

class MLPModel:
    def __init__(self, path):
        self.model = torch.jit.load(path)
        self.model.eval()

    def predict_mlp(self, text):
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(text)
        
        # Interpret the result
        probability = prediction.item()
        result = "Fog" if probability >= 0.5 else "No Fog"
        probability_percent = probability * 100  # Convert to percentage
        
        return {"probability of fog": f"{probability_percent:.4f}%", "prediction": result}