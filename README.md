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
