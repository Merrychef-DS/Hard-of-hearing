# **Hard of Hearing: Sound Classification Guide**

## **Overview**

This guide provides step-by-step instructions for setting up and running a sound classification system that listens for specific sounds in real-time and triggers alerts when the target sound ("eikon") is detected. It also covers prerequisites for accurate sound detection and instructions for using the dataset and model.

---

## **Step-by-Step Instructions**

### **1. Install Required Dependencies**
Before running the script, ensure that all required dependencies are installed. Use the `pip` package manager or the `requirements.txt` file included in your project directory. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

This will automatically install all the dependencies needed for the sound classification script.

### **2. Run the Sound Classification Script**
Once the dependencies are installed, execute the sound classification script to begin real-time sound detection. Run the following command in your terminal:

```bash
python audio_classification.py
```

The script will continuously monitor sounds in the environment. When the target sound ("eikon") is detected, it will trigger an alert.

## **Pre-Requisites for Accurate Detection**
To ensure optimal performance of the sound classification system, follow these best practices:

#### 1. Microphone Placement
Place the microphone close to the oven (or the source of the target sound).
This ensures that the microphone captures the target sound clearly and minimizes interference from background noise.

#### 2. Microphone Quality
Use a high-quality microphone capable of capturing the required sound frequencies with precision.
A good microphone ensures clear and consistent input, which is critical for accurate classification by the detection script.

## **Dataset and Model Details**

#### **1. Dataset Location**
The dataset used for training and testing the model is located at the following path:
```bash
"R:\K_Archive_\deaf starsbucks R&D\END OF COOKCYCLE DATASET"
```
This dataset contains the necessary sound data for detecting the target sound.

#### **2. Pre-Trained Model**
The project includes a pre-trained model saved in a .h5 file. This model can be used directly for sound classification.

However, if you need to update the model using new data, follow these steps to retrain it:

1. Add the updated data to the dataset directory.

2. Run the training script with the following command:

```bash
python Sound_train.py
```
This will train a new model based on the updated dataset and save it as a .h5 file.

## **Summary**
By following this guide, you can successfully set up and use a sound classification system for real-time sound detection. Make sure to use a high-quality microphone and place it appropriately to ensure accurate results. For further customization, the dataset and model can be updated and retrained as needed.

