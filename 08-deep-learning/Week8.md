# 📘 Fashion Image Classification with CNNs (ML Zoomcamp Session 8)

## 🌟 Context and Aim

This module focuses on building a deep learning model to classify clothing item images into one of 10 predefined categories (e.g., t-shirts, pants, skirts, etc.). The key application scenario is an online classifieds platform where users upload a clothing photo, and the system predicts its category automatically. The project introduces convolutional neural networks (CNNs), data augmentation, transfer learning, and evaluation metrics using TensorFlow and Keras. It marks a transition from tabular to image data in ML Zoomcamp.

---

## 🎩 Session-by-Session Breakdown

### 📺 8.1 – Fashion Classification

#### ✅ Goal

Introduce the image classification problem and define the real-world use case: clothing category prediction from an image using a CNN model.

#### 🧠 Concept

We switch from tabular to image data. Each image is a 3D array (height, width, RGB channels). The goal is to predict which of 10 categories an image belongs to. We use a subset of a fashion dataset, organized by folders per class. CNNs are used to process images due to their spatial hierarchy-capturing abilities. TensorFlow and Keras manage model building and training.

#### 🧠 Key Points Learned

* Introduction to the project use-case: auto-tagging clothes in online listings.
* The dataset is structured by folders (1 folder per class).
* Neural networks process images by learning spatial filters.
* Emphasis on practicality over deep theoretical exposition.

#### 🛠️ Tasks

* Download and explore the image dataset.
* Visualize training images.
* Understand problem framing (multi-class classification).
* Review class folder structure (e.g., "t-shirt", "pants").

---

### 📺 8.2 – TensorFlow and Keras

#### ✅ Goal

Set up TensorFlow/Keras for model development and image preprocessing.

#### 🧠 Concept

TensorFlow is a deep learning framework developed by Google. Keras is its high-level API that simplifies neural network creation and training. This session covers installing TensorFlow, importing necessary modules, and loading images using `tf.keras.preprocessing.image.load_img`. It also shows how to resize and convert images to NumPy arrays for model input.

#### 🧠 Key Points Learned

* TensorFlow is the backend engine, while Keras is the frontend API.
* Images need to be resized (e.g., 150x150 or 299x299) before feeding into a model.
* Images are internally represented as NumPy arrays with shape `(height, width, 3)`.
* Pixel values are integers from 0 to 255 per RGB channel.

#### 🛠️ Tasks

* Install and import TensorFlow/Keras.
* Load images from disk.
* Resize and visualize images.
* Convert PIL images to NumPy arrays.

---

### 📺 8.3 – Pre-Trained CNNs

#### ✅ Goal

Introduce the concept of transfer learning using pre-trained CNN models.

#### 🧠 Concept

Pre-trained convolutional neural networks like Xception or ResNet are trained on large datasets like ImageNet. They can be reused to extract image features (vector embeddings) without training from scratch. These embeddings can be passed to new dense layers for custom classification tasks.

#### 🧠 Key Points Learned

* Transfer learning saves time and resources.
* Pre-trained models provide a good feature extractor.
* You only train the final classification layer(s).
* Xception is used here as the backbone feature extractor.

#### 🛠️ Tasks

* Load a pre-trained Xception model (excluding top classifier layer).
* Pass resized images through it to extract features.
* Store image embeddings for later use.

---

### 📺 8.4 – CNN Theory

#### ✅ Goal

Explain how CNNs work, especially convolutional and dense layers.

#### 🧠 Concept

CNNs are composed of convolutional layers that detect spatial patterns in images using filters. These layers produce feature maps. As we stack more layers, the filters become sensitive to increasingly complex features (e.g., edges → shapes → objects). Dense layers then convert the extracted features into final predictions.

#### 🧠 Key Points Learned

* Filters slide over images to produce feature maps (activations).
* Convolutional layers stack hierarchically.
* Dense layers take the final feature vector and make a prediction.
* Final layer uses softmax for multi-class output.
* Intermediate layer weights are learned during training.

#### 🛠️ Tasks

* Visualize CNN layers and feature maps.
* Understand max pooling and dimensionality reduction.
* Understand vector embeddings and fully connected layers.

---




## 📺 **Session 8.5 – Transfer Learning**
### 🎯 **Context and Aim**

This session introduces *transfer learning*, a method for reusing pre-trained convolutional neural networks (CNNs) to solve a new image classification problem more efficiently. Instead of training a deep neural network from scratch—which is computationally expensive and data-hungry—we leverage a model previously trained on a large dataset (e.g., ImageNet) to extract meaningful features, and then train only a few new layers specific to our problem. This approach drastically reduces training time, avoids overfitting, and improves performance when limited data is available.

The practical focus is on implementing transfer learning using Keras in TensorFlow to classify clothing images into 10 categories using a small custom dataset.

### ✅ **Goal**

The objective of this session is to demonstrate how to build an image classification model using transfer learning with a pre-trained CNN (specifically, *Xception* trained on ImageNet) in Keras. The session walks through:

* Preprocessing and loading image data using `ImageDataGenerator`
* Loading a pre-trained CNN model and removing its top (dense) layers
* Adding new dense layers tailored to the new classification task
* Training only the new layers while keeping the convolutional base frozen
* Visualizing model performance across training epochs

---

### 🧠 **Concept**

Transfer learning refers to the process of taking a model trained on one task (e.g., ImageNet classification) and adapting it to a different but related task. The core idea is that early layers in a CNN learn generic features like edges, colors, and textures, which are useful across many vision tasks. Instead of re-learning these, we freeze the convolutional base and append new layers trained on our custom dataset.

The session uses the *Xception* model with `include_top=False` to exclude its fully connected layers. A *GlobalAveragePooling2D* layer is added to flatten the feature maps into vectors, followed by a new dense layer with a softmax output for 10 clothing categories. This pipeline enables efficient training with a relatively small dataset while achieving high performance.

Keras makes this process easy with modular APIs, including the `Model` class for functional model building, `ImageDataGenerator` for preprocessing, and `model.fit()` for training.

---

### 🧠 **Key Points Learned Here**

* **Why Transfer Learning?**
  Training a CNN from scratch requires millions of images. Transfer learning allows us to reuse features from a large dataset (e.g., ImageNet) and adapt them to smaller, task-specific datasets by training only the top layers.

* **Model Architecture with Pre-trained Base:**

  * Pre-trained CNNs like *Xception* are loaded with `include_top=False` to discard the dense output layers tailored to ImageNet.
  * Only the convolutional part is reused, and it is *frozen* (i.e., not trainable).
  * On top of the frozen base, a `GlobalAveragePooling2D` layer reduces the dimensionality.
  * Finally, a new dense layer with 10 outputs (for the 10 classes) is trained.

* **Data Preparation:**

  * Images are loaded using `ImageDataGenerator`, resized to 150x150x3 to reduce training cost.
  * `flow_from_directory()` automatically labels images based on folder structure and returns batches.
  * One-hot encoding is used for multi-class classification.

* **Training Configuration:**

  * Optimizer: Adam with a custom learning rate (0.01).
  * Loss: `CategoricalCrossentropy(from_logits=True)` because the final dense layer does **not** use softmax (it outputs raw logits).
  * Metric: Accuracy.

* **Training Loop and Evaluation:**

  * The model is trained using `.fit()` with training and validation generators.
  * Accuracy improves rapidly in the first few epochs.
  * Model begins overfitting after \~4 epochs (training accuracy \~99%, validation \~80%).

* **Performance Tracking:**

  * A `History` object stores per-epoch accuracy and loss.
  * Matplotlib is used to visualize accuracy and loss curves over epochs.

---

### 🛠️ **Tasks**

1. **Preprocess Dataset**

   * Load images using `ImageDataGenerator`.
   * Normalize using `preprocess_input` from the Xception model.

2. **Create Data Generators**

   * Use `flow_from_directory()` for both training and validation sets.
   * Set image size to 150x150 and batch size to 32.
   * Disable shuffling for validation set.

3. **Build Transfer Learning Model**

   * Load `Xception` with `weights='imagenet'`, `include_top=False`.
   * Freeze the base model with `trainable=False`.
   * Add `GlobalAveragePooling2D()` layer to flatten features.
   * Add `Dense(10)` layer (no activation) for classification logits.

4. **Compile Model**

   * Use `Adam(learning_rate=0.01)`.
   * Use `CategoricalCrossentropy(from_logits=True)`.
   * Track `accuracy` as a metric.

5. **Train Model**

   * Use `model.fit()` for 10 epochs.
   * Monitor training and validation accuracy/loss.
   * Store result in `history`.

6. **Plot Accuracy**

   * Extract training and validation accuracy from `history.history`.
   * Plot to detect overfitting (training acc > validation acc).



### 📺 8.6 – Training a Simple Model

#### ✅ Goal

Build and train a basic CNN on the image data.

#### 🧠 Concept

A simple CNN consists of a few convolutional + pooling layers, followed by a dense layer. This architecture is used to establish a baseline before using transfer learning.

#### 🧠 Key Points Learned

* A basic CNN can already perform well with enough training data.
* Use ReLU activations and `softmax` in the final layer.
* Compile with `categorical_crossentropy` and `Adam` optimizer.
* Track training vs. validation loss and accuracy.

#### 🛠️ Tasks

* Define and compile a sequential Keras CNN.
* Train the model using `.fit()`.
* Plot training history.
* Save model and weights.

---

### 📺 8.7 – Evaluating the Model

#### ✅ Goal

Assess the model’s performance and identify potential issues.

#### 🧠 Concept

Evaluation is done via accuracy and loss on the test dataset. Visualization of misclassified images is used to understand model limitations.

#### 🧠 Key Points Learned

* Always use a separate test set for final performance assessment.
* Confusion matrix reveals which classes are confused.
* Misclassifications often come from ambiguous or poor-quality images.

#### 🛠️ Tasks

* Use `.evaluate()` on test data.
* Plot confusion matrix using `sklearn.metrics`.
* Display a few misclassified images and their predicted/true labels.

---

### 📺 8.8 – Dropout

#### ✅ Goal

Prevent overfitting using dropout regularization.

#### 🧠 Concept

Dropout randomly deactivates neurons during training, which helps prevent the network from relying too heavily on specific features.

#### 🧠 Key Points Learned

* Dropout is a form of regularization.
* Typical dropout rates: 0.3 to 0.5 for dense layers.
* Only active during training, not inference.
* Should be added before fully connected layers.

#### 🛠️ Tasks

* Add `Dropout()` to your model architecture.
* Retrain and compare overfitting behavior.
* Observe validation accuracy improvements.

---

### 📺 8.9 – Learning Rate Scheduling

#### ✅ Goal

Optimize training using learning rate adjustment strategies.

#### 🧠 Concept

Choosing the right learning rate is crucial. Learning rate schedules adapt it dynamically to improve training convergence and stability.

#### 🧠 Key Points Learned

* Too high → divergence; too low → slow convergence.
* `ReduceLROnPlateau` lowers learning rate if validation stagnates.
* Learning rate is one of the most important hyperparameters.

#### 🛠️ Tasks

* Integrate `ReduceLROnPlateau` callback in `.fit()` call.
* Monitor learning rate changes and impact on loss curves.
* Experiment with different `factor` and `patience` values.






## 📺 \[Session 8.10 – Data Augmentation]

### ✅ Goal

The objective of this session is to improve the **generalization ability** of the image classification model by using **data augmentation** techniques. Instead of training the model repeatedly on the same images, the idea is to generate **new, varied images** from the existing dataset using simple transformations (e.g., flipping, rotation, shifting). This combats overfitting, encourages the model to focus on meaningful patterns, and prepares it to handle real-world variations that might appear in unseen data. Data augmentation becomes particularly valuable when working with **small datasets**, helping simulate diversity without collecting new data.

---

### 🧠 Concept

**Data augmentation** is a form of regularization where new, synthetic training examples are created from existing ones using random but label-preserving transformations. In image classification, common augmentations include flipping, rotating, shifting, zooming, shearing, adjusting brightness/contrast, or even randomly covering parts of the image.

These transformations are applied on-the-fly during training using tools like Keras’s `ImageDataGenerator`. For instance, rotating an image ±30°, flipping it vertically, or zooming 10% in/out can generate a “new” training example, even though the label remains unchanged.

This session explains that **not all augmentations make sense for every dataset**. For the clothing dataset used here, vertical flips and zoom worked best, while rotation and shifting degraded performance. The instructor emphasizes that augmentation is a **hyperparameter**, and its benefit must be empirically validated—just like dropout rate or layer size. Importantly, augmentations are only applied to the training data—not to validation or test sets—to maintain consistency in evaluation.

---

### 🧠 Key Points Learned Here

* **Data augmentation** helps neural networks generalize better by presenting different versions of the same image across epochs.
* Common augmentations include: `rotation_range`, `width_shift_range`, `height_shift_range`, `zoom_range`, `shear_range`, `horizontal_flip`, and `vertical_flip`.
* Augmentations are **randomized**, so each epoch the model sees slightly different images.
* **Only the training set** is augmented—never the validation set—to ensure consistent evaluation.
* Over-augmentation (too strong transformations) can actually degrade model performance, as seen with high shear or rotation in this session.
* Some augmentations (like flipping) are **semantically safe** for symmetrical objects (e.g., shirts), but dangerous for others (e.g., digits or faces).
* GPU utilization may **drop** when using augmentations, since transformations are typically performed on the **CPU**, causing a bottleneck.
* The instructor observes that augmentation doesn’t always improve results—it’s an **experimental decision**, not a guaranteed gain.
* A model trained with simple vertical flip performed better than using complex augmentation combinations.
* Data augmentation can also be treated like any other hyperparameter and should be tuned and tested accordingly.

---

### 🛠️ Tasks

* Explained why and when to use data augmentation.
* Reviewed basic transformations visually (flip, shift, rotate, zoom, shear).
* Used Keras’s `ImageDataGenerator` to apply:

  * `rotation_range=30`
  * `width_shift_range=0.1`
  * `height_shift_range=0.1`
  * `zoom_range=0.1`
  * `shear_range=10`
  * `vertical_flip=True`
* Trained the model for 50 epochs with these augmentations.
* Observed and compared training/validation accuracy with and without augmentation.
* Discovered that **simple vertical flip** augmentation performed best in practice.
* Reflected on GPU inefficiencies caused by CPU-based preprocessing.
* Concluded that augmentation benefits vary by dataset and require empirical testing.
* Suggested exploring `tf.data` pipelines for more efficient data loading and augmentation in future work.




## 📺 Session 8.11 – Training a Larger Model (Xception 299×299)

### ✅ Goal

The objective of this session is to **train a more accurate neural network** by scaling up the input image size from 150×150 to 299×299 and using the full capacity of the pre-trained Xception model. This transition aims to enhance model performance by leveraging more detailed features in higher-resolution images. Additionally, techniques like **data augmentation**, **dropout**, and **smaller learning rates** are re-evaluated to reduce overfitting and improve generalization on the validation set.

---

### 🧠 Concept

In earlier lessons, a smaller version of the Xception model was used (150×150 inputs) for faster experimentation. Here, the **full-resolution version (299×299)** is employed, allowing the convolutional layers to extract **richer, finer-grained features** from the input. However, this increases training time by about 4×.

The model uses **transfer learning** with frozen convolutional base and trains a classification head. To combat **overfitting**, **data augmentation** (e.g., zooming, flipping) and **dropout** are used. Additionally, a **smaller learning rate** is applied to stabilize training and reduce validation loss oscillations.

Checkpointing is introduced to save the best performing model based on validation accuracy. This larger model is expected to better capture intricate clothing features, increasing accuracy while preserving generalizability.

---

### 🧠 Key Points Learned Here

* **Larger images (299×299)** provide more information for the model but significantly increase computation time (\~4× slower per step).
* **Training accuracy** tends to grow quickly on large models, leading to potential overfitting if not regulated.
* **Data augmentation** (e.g., zoom, vertical flip) helps reduce overfitting by showing the model new variations of the same data at each epoch.
* **Validation accuracy** becomes more reliable with augmentation since the model no longer memorizes exact inputs.
* Adding **model checkpointing** is essential when training large models to avoid losing the best version.
* **Lower learning rates** smooth out the learning curve and prevent the model from diverging or oscillating during training.
* Even though augmentation increases CPU load and slows training, it can produce models that generalize better and have lower validation gaps.

---

### 🛠️ Tasks

* Modified the `input_size` parameter to allow flexible resolution (299×299 for Xception).
* Reused and adapted the model code from previous sessions with added configurability.
* Integrated **data augmentation** (zoom range, vertical flip) into the training pipeline.
* Enabled **model checkpointing** to store the best-performing model.
* Switched between augmentation on/off to assess impact on validation accuracy.
* Compared training/validation accuracies with and without augmentation.
* Reduced the **learning rate** to smooth the training process and improve generalization.
* Evaluated overfitting risk by analyzing accuracy gaps and confirmed that **augmented models** yielded more reliable results.

---
Voici le résumé détaillé demandé pour la Session 8.12 du ML Zoomcamp :

---

## 📺 Session 8.12 – Model Evaluation & Prediction

### ✅ Goal

The objective of this session is to **evaluate the performance of the final trained model on a separate test dataset** and demonstrate how to **load the model and use it to make predictions** on individual images. This confirms whether the model generalizes well and provides a practical example of how to use a trained model in real-world applications like clothing classification. The session also includes post-processing the output logits to determine the predicted class.

---

### 🧠 Concept

After training a high-performing convolutional neural network (based on the Xception architecture with 299×299 input size), it is essential to **validate its performance on unseen test data**. This is done by loading the saved model checkpoint and applying it to the test dataset using the `.evaluate()` method, which returns both the loss and accuracy. Furthermore, this session introduces the process of making **individual predictions** on images by loading an image, resizing it, converting it to a NumPy array, applying the correct preprocessing function, and then passing it through the model using `.predict()`. The resulting logits (unnormalized scores) are mapped to class labels, and the top prediction is extracted to interpret the result. This full pipeline simulates a production use-case for image classification.

---

### 🧠 Key Points Learned Here

* **Model Loading**: The model trained and saved in the previous session is loaded using `keras.models.load_model()`.
* **Test Evaluation**: Using `.evaluate()`, we assess test performance, confirming the model has not overfit (test accuracy \~90%).
* **Image Preprocessing**: New images must be resized (299×299), converted to arrays, and passed through the same preprocessing pipeline used during training (`preprocess_input` from Xception).
* **Prediction Pipeline**: After preprocessing, the image is passed through the model with `.predict()`, yielding **logits** for each class.
* **Interpreting Results**: Logits are matched to class labels using `zip()` and converted to a dictionary for readability. Although not converted to probabilities, the relative values indicate class likelihoods.
* **Correct Classification**: The model correctly classifies a test image (e.g., pants), showing strong performance and robustness.
* **Practical Readiness**: This session bridges training and real-world usage, showing how to package a trained model into a functional prediction tool.

---

### 🛠️ Tasks

* Loaded a pre-trained model using `keras.models.load_model("model_name")`
* Loaded and prepared the test dataset (ImageDataGenerator + `flow_from_directory`)
* Evaluated the model on the test dataset using `.evaluate()`
* Loaded a single image with `load_img()` and resized it to 299×299
* Converted the image to a NumPy array and applied Xception preprocessing
* Used `.predict()` to obtain logits from the model
* Created a mapping between predicted scores and class labels using `zip()` and dictionary construction
* Interpreted and displayed the top prediction
* Verified the model generalizes well to unseen images

---



Voici le résumé structuré de la **Session 8.13** du ML Zoomcamp — une session de conclusion riche et récapitulative :

---

## 📺 Session 8.13 – Summary of Image Classification with CNNs

### ✅ Goal

The goal of this final session is to **summarize everything learned in Session 8** of the ML Zoomcamp, which focused on solving an image classification task using convolutional neural networks (CNNs). The case study involved building a web-based system where a user uploads an image of a clothing item, and the system predicts its category among 10 predefined classes. Throughout the session, learners explored data loading, transfer learning, architecture tuning, regularization techniques, model evaluation, and prediction pipelines. This session also introduces **ideas for further exploration**, such as alternative model architectures, augmentation libraries, new datasets, and using frameworks like PyTorch. It aims to consolidate understanding and open doors to more advanced experimentation or deployment scenarios.

---

### 🧠 Concept

This session wraps up the process of applying deep learning to image classification using **Keras and TensorFlow**. The core idea was to **leverage transfer learning** by using a pre-trained CNN (Xception) as a fixed feature extractor and customizing the output layers to fit a new classification task. The pipeline included data preprocessing, model construction, tuning hyperparameters (like learning rate and dense layer size), and implementing regularization through dropout and data augmentation.

The model was evaluated on separate validation and test sets, confirming it generalized well (\~90% accuracy). Students also learned how to apply the trained model to individual images for real-world inference. Beyond training, the session highlighted **practical enhancements**: experimenting with other CNN architectures (e.g., ResNet, MobileNet), trying other deep learning libraries (PyTorch, MXNet), and using creative datasets (e.g., hotdog vs. not hotdog). The session bridges theoretical understanding, practical implementation, and ideas for real-world deployment.

---

### 🧠 Key Points Learned Here

* **Project Overview**: Classify uploaded clothing images into 10 categories using a convolutional neural network.
* **Frameworks Used**: Built the pipeline using TensorFlow and Keras, abstracting away low-level operations to focus on high-level design.
* **Transfer Learning**: Reused a pre-trained Xception model trained on ImageNet. Retained convolutional layers and replaced the classifier head.
* **Model Tuning**:

  * Explored different **learning rates** (fast vs. stable convergence).
  * Added **intermediate dense layers** to learn richer representations.
  * Used **dropout** to reduce overfitting by randomly deactivating neurons during training.
  * Applied **data augmentation** to generate synthetic image variations and improve generalization.
* **Training Strategy**:

  * Started with 150×150 images for fast prototyping.
  * Finished with full-size 299×299 images for final model training.
  * Used **checkpointing** to save the best model automatically.
* **Evaluation**:

  * Achieved 90% accuracy on both validation and test datasets.
  * Validated the model on unseen data to confirm robustness.
* **Prediction Pipeline**:

  * Loaded external images.
  * Preprocessed them identically to training data.
  * Passed through the model and decoded the logits to get predicted classes.
* **Next Steps Suggested**:

  * Try other architectures: MobileNet (faster), ResNet50 (lighter), Inception-ResNet (more accurate but slow).
  * Use different frameworks (e.g., PyTorch).
  * Experiment with other datasets like hotdog/not hotdog, Avito classifieds.
  * Use augmentation libraries like **Albumentations** for more advanced transforms.
