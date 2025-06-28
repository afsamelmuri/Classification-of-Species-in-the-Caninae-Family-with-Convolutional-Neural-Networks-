# 🐾 Canidae Species Classification using ResNet50 and Transfer Learning

## 📌 Abstract
Species classification is essential in wildlife protection, ecological research, and livestock monitoring. Manual identification is time-consuming, error-prone, and not scalable. This study introduces a deep learning-based approach using **ResNet50 with transfer learning**, targeting the classification of four closely related **Canidae species**: **Dog, Wolf, Fox, and Hyena**.

The model achieved a **91.4% test accuracy**, employing **extensive data augmentation** to handle diverse environmental conditions (lighting, background, pose). Evaluation metrics include **accuracy**, **precision**, **recall**, and **F1-score**, demonstrating reliable classification capabilities suitable for ecological research and conservation.

**Keywords:** Species classification · Wildlife conservation · Deep learning · CNNs · ResNet · Transfer learning · Data augmentation

---

## 📖 1. Introduction

Biodiversity forms the core of ecosystem stability. However, habitat destruction and urban sprawl threaten numerous species. Manual classification methods are inefficient for large-scale monitoring. **Deep learning**, particularly **CNNs**, offers automated and scalable solutions.

The **Canidae family**—comprising dogs, wolves, foxes, and hyenas—presents identification challenges due to overlapping appearances and behaviors. Accurate classification of these species is vital for ecological balance, conservation, and wildlife tracking.

This project aims to design an efficient, automated image classifier using a **pretrained ResNet50 model**, fine-tuned for this specific task, and augmented to simulate real-world conditions.

---

## 🧪 2. Methodology

### 2.1 Model Overview

- **Backbone Model**: Pretrained **ResNet-50**  
- **Modification**: Final layer replaced with a 4-class fully connected output  
- **Optimizer**: Adam (lr=0.001)  
- **Loss Function**: CrossEntropyLoss  
- **Training Strategy**: Freeze base layers, train FC layer  
- **Augmentation**: Random flips, rotations, brightness adjustments, resizing

---

## 📷 3. Data Collection & Preprocessing

- **Classes**: Dog, Fox, Hyena, Wolf  
- **Total Images**: **7,905**
  - **Training**: 5,121  
  - **Validation**: 1,085  
  - **Testing**: 1,699  

| Class     | Train | Validation | Test | Total |
|-----------|-------|------------|------|-------|
| Dog       | 1400  | 300        | 300  | 2000  |
| Fox       | 1400  | 300        | 800  | 2500  |
| Hyena     | 921   | 185        | 300  | 1406  |
| Wolf      | 1400  | 300        | 299  | 1999  |

📌 **Preprocessing Steps**:
- Resizing images to 224x224
- Normalizing using ImageNet means and std
- Data balancing ensured through sampling

---

### 🧠 4. Model Selection

We use **ResNet50**, a 50-layer deep convolutional neural network, pre-trained on ImageNet. Using **transfer learning**, the original fully connected layer (1000-class) was replaced by a **custom FC layer with 4 outputs**. This preserves learned low-level features while adapting to our custom dataset.

> 📈 Benefits of Transfer Learning:
> - Faster convergence
> - Improved accuracy
> - Efficient training on small/medium datasets

---

## ⚙️ 5. Model Training

**Training Setup:**
- **Epochs**: 4  
- **Optimizer**: Adam (lr=0.001)  
- **Frozen Layers**: All ResNet base layers  
- **Trained Layers**: Custom FC layer only

### Training Flow:
1. Forward pass through frozen layers and FC layer
2. CrossEntropyLoss used to compare predictions with labels
3. Backpropagation through FC layer only
4. Weight updates via Adam optimizer

---

## 📐 6. Evaluation Metrics

| Metric       | Formula                                |
|--------------|-----------------------------------------|
| Accuracy     | (TP + TN) / (TP + TN + FP + FN)         |
| Precision    | TP / (TP + FP)                          |
| Recall       | TP / (TP + FN)                          |
| F1-Score     | 2 × (Precision × Recall) / (Precision + Recall) |

---

## 📊 7. Results and Discussion

The model showed **strong performance across all species**, achieving a **91.4% final test accuracy**. Below is the detailed per-class performance:

### 🔹 Table 1: Final Per-Class Metrics

| Species | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Dog     | 98.33%   | 95.19%    | 92.33% | 93.74%   |
| Fox     | 89.33%   | 84.02%    | 94.67% | 89.03%   |
| Hyena   | 93.33%   | 97.51%    | 94.32% | 94.32%   |
| Wolf    | 89.99%   | 90.64%    | 89.14% | 89.14%   |

### 🔹 Table 2: Epoch-Wise Accuracy

| Epoch | Validation Accuracy | Loss  |
|-------|----------------------|-------|
| 1     | 91.08%               | 0.41  |
| 2     | 91.96%               | 0.35  |
| 3     | 92.32%               | 0.31  |
| 4     | 92.58%               | 0.28  |

### 🔹 Table 3: Sample Predictions (Accuracy by Class)

| Ground Truth | Predicted | Accuracy |
|--------------|-----------|----------|
| DOG          | DOG       | 99.82%   |
| FOX          | FOX       | 99.99%   |
| HYENA        | HYENA     | 99.74%   |
| WOLF         | WOLF      | 99.59%   |

---

## 🧾 8. Project Structure


## 🛠️ 9. How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/yourusername/canidae-classification.git
cd canidae-classification

## 🔬 3. Methodology

The goal of this study is to accurately classify four Canidae species—**Dog, Wolf, Fox, and Hyena**—using a deep learning-based image classification approach. The methodology is structured in four main phases: **Data Collection & Preprocessing**, **Model Selection**, **Model Training**, and **Evaluation & Metrics**. Each component is carefully designed to ensure high accuracy, robustness to environmental variability, and practical applicability in ecological research.

---

### 📁 3.1 Data Collection and Preprocessing

To train the model, we compiled a custom dataset comprising **7,905 labeled images** across the four target species. The dataset was split into three subsets to ensure balanced evaluation and model validation:

| Species | Training | Validation | Testing | Total  |
|---------|----------|------------|---------|--------|
| Dog     | 1,400    | 300        | 300     | 2,000  |
| Fox     | 1,400    | 300        | 800     | 2,500  |
| Hyena   |   921    | 185        | 300     | 1,406  |
| Wolf    | 1,400    | 300        | 299     | 1,999  |

> Total: 7,905 high-quality, labeled images  
> Input size: 224x224 pixels, 3-channel RGB

**Preprocessing Steps:**
- **Resizing** all images to **224×224 pixels** to meet the input requirement of ResNet50.
- **Normalization** using ImageNet mean and standard deviation values.
- **Augmentation Techniques** to simulate real-world scenarios:
  - Random horizontal/vertical flips
  - Rotations (±30°)
  - Color jitter (brightness, contrast, saturation)
  - Random crops and affine transforms
  - Background variation simulation to improve generalization

These steps improve the model’s ability to generalize to varied ecological environments, camera angles, lighting conditions, and animal poses.

---

### 🧠 3.2 Model Selection

To tackle the classification problem efficiently, we chose the **ResNet50** deep convolutional neural network due to its proven performance in complex visual tasks.

**Why ResNet50?**
- 50-layer deep residual network.
- Handles vanishing gradient problem via **residual (skip) connections**.
- Trained on ImageNet (over 14 million images).
- Extracts hierarchical and abstract features from images.

#### 🔁 Transfer Learning Strategy

- We **froze** all the convolutional base layers of the pretrained ResNet50 model.
- Replaced the **original fully connected (FC) layer** with a new FC layer consisting of **4 output neurons** (one for each class).
- Only the newly added FC layer was trained on the Canidae dataset.
- This allowed:
  - Faster convergence
  - Lower computational cost
  - Preservation of low-level features (edges, textures, shapes) learned from ImageNet

**Architecture Overview:**

```text
Input Image → ResNet50 Base (frozen) → New FC Layer (4 outputs) → Softmax → Prediction
