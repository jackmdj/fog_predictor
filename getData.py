import requests
from datetime import datetime
import os
import time
from bs4 import BeautifulSoup
import re
import csv
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def download_image(x_times):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0'
    }
    image_url = 'https://www.805webcams.com/cameras/cachumalake/camera.jpg'
    base_dir = os.getcwd()
    pred_dir = os.path.join(base_dir, 'Images', 'Prediction', 'train')
    for i in range(x_times):
        try:
            image_response = requests.get(image_url, headers=headers, stream=True)
            if image_response.status_code == 200:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{current_time}.jpg"
                full_path = os.path.join(pred_dir, filename)
                
                with open(full_path, 'wb') as f:
                    for chunk in image_response:
                        f.write(chunk)
                print(f"Image downloaded successfully and saved as {full_path}!")
            else:
                print(f"Failed to download the image. Status code: {image_response.status_code}")
            if x_times > 1:
                time.sleep(300)  # Delays for 5 minutes
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

def scrape_weather_info():
    url = 'https://www.localconditions.com/weather-lake-cachuma-california/93464/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0'
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
                
            # Find the element by ID
            weather_text_element = soup.find(id='readLCWXtxt')
            if weather_text_element:
                return weather_text_element.get_text(strip=True)
            else:
                print("Weather information element not found.")
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def write_to_csv(data, filename='prediction_data.csv'):
    fieldnames = ['Observation Time', 'Temperature', 'Humidity', 'Dew Point', 'Trend', 'Fog']
    
    # Check if file exists to determine if we need to write headers
    try:
        with open(filename, 'x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dict(zip(fieldnames, data)))
    except FileExistsError:
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(dict(zip(fieldnames, data)))

def parse_weather_report(text):
    # Regular expressions to find relevant data and observation time
    temperature_pattern = r"It is (\d+) degrees fahrenheit"
    humidity_pattern = r"humidity is (\d+\.?\d*) percent"
    dew_point_pattern = r"dew point of (\d+\.?\d*) degrees fahrenheit"
    trend_pattern = r"that is (\w+) since the last report"
    
    # Extract data using regular expressions
    temperature = re.search(temperature_pattern, text)
    humidity = re.search(humidity_pattern, text)
    dew_point = re.search(dew_point_pattern, text)
    trend = re.search(trend_pattern, text)
    
    # Record time and date of observation
    observation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a list to store the data
    data = [observation_time]

    data.append(temperature.group(1) + ' °F')
    data.append(humidity.group(1) + ' %')
    data.append(dew_point.group(1) + ' °F')
    data.append(trend.group(1))

    # Add an empty column for 'Fog' as a binary
    data.append('')  # Empty string for manual update
    
    return data

def read_most_recent_data(filename='prediction_data.csv'):
    # Using pandas to read the last row of the CSV
    df = pd.read_csv(filename)
    most_recent_entry = df.iloc[-1]
    return most_recent_entry

def WeatherScript(x_times):
    for i in range(x_times):
        parsed_data=parse_weather_report(scrape_weather_info())
        write_to_csv(parsed_data)
        print("Data has been written to CSV.")
        if x_times > 1:
            time.sleep(300)  # Delays for 5 minutes

def get_Data(x_times): # 24 for 2 hours
    for i in range(x_times):
        WeatherScript(1)
        download_image(1)
        if x_times > 1:
            time.sleep(60)

def calculate_mean_std(loader):
    # Variances and means across channels
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0
    
    # Calculate mean
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    
    # Calculate std
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).mean(2).sum(0)
    
    std = torch.sqrt(std / nb_samples)
    return mean, std

def calcNormalization(batch_size):
    # Define transformations, adjust according to your need
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all images to 256x256
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    # Get the path to the current working directory
    base_dir = os.getcwd()
    pred_dir = os.path.join(base_dir, 'ImageData', 'train')

    # Loading the data
    pred_data = datasets.ImageFolder(root=pred_dir, transform=transform)
    pred_loader = DataLoader(pred_data, batch_size=batch_size)
    mean, sd = calculate_mean_std(pred_loader)
    return mean, sd

def transform():
    mean, sd = calcNormalization(22)
    base_dir = os.getcwd()
    pred_dir = os.path.join(base_dir, 'Images', 'Prediction')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all images to 256x256
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=mean.tolist(), std=sd.tolist())
    ])
    pred_data = datasets.ImageFolder(root=pred_dir, transform=transform)
    image, _ = pred_data[-1]  # dataset returns (image, label)
    return image

def read_most_recent_stats(filename='weather_data.csv'):
    data = pd.read_csv(filename, encoding='ISO-8859-1')
    
    # Convert features to numeric values
    data['Temperature'] = data['Temperature'].str.extract(r'(\d+)').astype(float)
    data['Dew Point'] = data['Dew Point'].str.extract(r'(\d+)').astype(float)
    data['Humidity'] = data['Humidity'].str.replace('%', '').astype(float)
    
    # Encode categorical data
    trend_mapping = {'steady': 2, 'rising': 1, 'falling': 0}
    data['Trend'] = data['Trend'].map(trend_mapping).astype(float)
    

    
    # Convert to a numpy array and then to a PyTorch tensor
    datapoint = torch.tensor(data[['Temperature', 'Humidity', 'Dew Point', 'Trend']].values, dtype=torch.float32)
    
    return datapoint[-1, :]

def read_data():
    get_Data(1)
    image = transform()
    stats = read_most_recent_stats()
    return image, stats