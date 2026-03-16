# Smart Mask Detection and Temperature Monitoring System

A real-time computer vision and embedded system that detects whether a person is wearing a face mask and simultaneously measures their body temperature using a non-contact infrared sensor.

This project integrates **Deep Learning**, **Computer Vision**, and **Embedded Systems** to create a real-time safety monitoring solution.

---

# Project Overview

This system performs two primary tasks simultaneously:

1. Detects whether a person is wearing a face mask using a deep learning model.  
2. Measures the person's temperature using an MLX90614 infrared temperature sensor connected to a microcontroller.

The results are displayed live through a webcam feed where the system overlays:

- Mask detection result  
- Confidence score  
- Temperature reading  
- Fever warning if temperature exceeds threshold

This type of system can be used for:

- Hospital entrances  
- Office buildings  
- Airport screening  
- Public safety monitoring

---

# System Demo

Below is a demo of the system detecting a face mask and displaying the temperature, click on the demo to see the longer version.

[![Watch Demo](demo.gif)](https://drive.google.com/file/d/1Le8rTDF2wX4iauqJGVIYcLUZ12OClhCC/view?usp=drive_link)

---

# Hardware Architecture

The temperature sensing part uses the **MLX90614 Non-contact IR Temperature Sensor** connected to an **Adafruit Metro 328 (ATmega328 microcontroller)**.

The microcontroller sends temperature readings to the Python program through **Serial Communication**.

---

# Circuit Diagram

The following diagram shows how the **MLX90614 temperature sensor** is connected to the microcontroller.

![Circuit Diagram showing MLX90614 connected to ATmega328](circuit_diagram.png)

**Figure 1** Circuit diagram: MLX90614 connected to Adafruit Metro 328 via I2C.


---

# Hardware Components

| Component | Description |
|--------|-------------|
| MLX90614 IR Sensor | Non-contact infrared temperature sensor |
| Adafruit Metro 328 | Microcontroller based on ATmega328 |
| Webcam | Used for face detection |
| Jumper Wires | Circuit connections |
| Computer | Runs Python and deep learning model |

---

# Sensor Pin Connections

| Sensor Pin | Microcontroller Pin |
|-----------|---------------------|
| VIN | 3.3V |
| GND | GND |
| SDA | SDA (I2C Data) |
| SCL | SCL (I2C Clock) |

---

# Software Architecture

The system consists of the following modules:

```
Step1:Camera Feed

Step2:Face Detection (OpenCV DNN)

Step3:Face Mask Classification (MobileNetV2)

Step4:Temperature Data (Serial from Arduino)

Step5:Overlay Results on Video Stream
```

---

# Deep Learning Model

The mask detection model is based on **MobileNetV2**, a lightweight convolutional neural network designed for real-time applications.

The model is trained using images from two classes:

- With Mask  
- Without Mask

### Training Pipeline

The training script:

- Loads dataset images  
- Applies preprocessing  
- Uses data augmentation  
- Trains MobileNetV2 head layers  
- Saves the trained model

Training script location:

```
training/train.py
```

---

# Dataset Structure

```
dataset/
│
├── with_mask
│   ├── image1.jpg
│   ├── image2.jpg
│
└── without_mask
    ├── image1.jpg
    ├── image2.jpg
```


# Installation

## Clone the Repository

```
git clone <https://github.com/Tanupvats/mask-temp-monitor.git>
cd mask-temp-monitor
```


# Install Dependencies

```
pip install -r requirements.txt
```
---
Dependencies include:

- OpenCV  
- TensorFlow  
- NumPy  
- PySerial

---

# Training the Mask Detection Model

Run the training script:

```
python train.py
```

This script:

1. Loads images  
2. Applies preprocessing  
3. Performs augmentation  
4. Trains the MobileNetV2 classifier  
5. Saves the model

Output:

```
custom_mask_detector.model
training_plot.png
```

---

# Evaluating the Model

Run:

```
python evaluate.py
```

This will generate:

- Classification report  
- Confusion matrix

Output file:

```
confusion_matrix.png
```

---

# Running the Real-Time System

Make sure the Arduino is connected and the serial port is correct.

Update the port in:

```
main.py
```

Example:

```
SERIAL_PORT = 'COM5'
```

Run the system:

```
python main.py
```

The system will:

1. Capture webcam video  
2. Detect faces  
3. Predict mask or no-mask  
4. Read temperature from sensor  
5. Display results on screen

---

# Serial Communication

The Python program receives temperature readings from the Arduino via serial communication.

Example data format:

```
TEMP:36.5
```

To test serial communication separately:

```
python runtime/test_serial.py
```

---

# Fever Detection Logic

The system flags fever if temperature exceeds **37.5°C**.

Example display:

```
Mask: 99.9% | Temp: 38.1°C (FEVER)
```

---

# Key Features

Real-time face detection  
Mask classification using deep learning  
Non-contact temperature sensing  
Serial communication with microcontroller  
Live monitoring interface  
Automatic fever detection

---

---

# Applications

Public health monitoring  
Hospital screening systems  
Airport checkpoints  
Office access control  
Smart building entry systems

---

# Future Improvements

Edge deployment using Raspberry Pi  
Integration with thermal cameras  
Cloud-based monitoring dashboard  
Automatic access control system

---

# Author

**Tanup Vats**


