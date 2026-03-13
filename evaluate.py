import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_pipeline(testX, testY, label_names=["with_mask", "without_mask"]):
    print("[*] Loading custom model...")
    model = load_model("custom_mask_detector.model")

    print("[*] Evaluating network...")
   
    predIdxs = model.predict(testX, batch_size=32)

    
    predIdxs = np.argmax(predIdxs, axis=1)
    trueIdxs = np.argmax(testY, axis=1)

    
    print("\n--- Classification Report ---")
    print(classification_report(trueIdxs, predIdxs, target_names=label_names))

    
    cm = confusion_matrix(trueIdxs, predIdxs)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png")
    print("[*] Saved confusion_matrix.png")

if __name__=='__main__':

    evaluate_pipeline(testX, testY)