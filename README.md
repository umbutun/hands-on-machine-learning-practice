# 🤖 Hands-On Machine Learning Practice

*A collection of practical notebooks, exercises, and projects from my journey through*  
**📘 “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” (Aurélien Géron)**

---

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)]()

---

### 🧭 Overview
This repository documents my **hands-on learning journey** as I work through *Hands-On Machine Learning* by Aurélien Géron.  
Each chapter folder includes:
- Annotated Jupyter notebooks  
- My own notes, experiments, and reflections  
- Small standalone projects inspired by the chapter content  

> 🚀 The goal is to learn deeply by building, testing, and explaining everything from scratch.

---

## 📚 Chapters & Notebooks

| Chapter | Title | Description | Notebook / Repo Link |
|:--------:|--------|--------------|-----------------------|
| 1 | The Machine Learning Landscape | Overview of core ML concepts, types of learning (supervised, unsupervised, reinforcement), and key challenges like overfitting and underfitting. | [View Notebook](./notebooks/chapter_01_the_machine_learning_landscape/chapter_01_notebook.ipynb) |
| 2 | End-to-End Machine Learning Project 🏠 | Practical case study: building a real estate price predictor from scratch. Covers data exploration, feature engineering, pipelines, and model evaluation. | [Separate Project Repo →](https://github.com/umbutun/real-estate-price-predictor) |
| 3 | Classification | Introduction to classification tasks. Includes logistic regression, precision/recall, ROC curves, and multiclass strategies using the MNIST dataset. | [View Notebook](./notebooks/chapter_03_classification/chapter_03_notebook.ipynb) |
| 4 | Training Models | In-depth look at linear models: linear regression, gradient descent, regularization (Ridge, Lasso, ElasticNet), and early stopping techniques. | [View Notebook](./notebooks/chapter_04_training_models/chapter_04_notebook.ipynb) |
| 5 | Support Vector Machines (SVMs) | Fundamentals of SVMs for linear and nonlinear classification and regression. Explains kernels, margins, and regularization parameters. | [View Notebook](./notebooks/chapter_05_support_vector_machines/chapter_05_notebook.ipynb) |
| 6 | Decision Trees | Decision tree construction, visualization, Gini impurity, entropy, and how trees handle non-linear datasets. | [View Notebook](./notebooks/chapter_06_decision_trees/chapter_06_notebook.ipynb) |
| 7 | Ensemble Learning and Random Forests | Combines multiple models to boost performance: bagging, boosting, stacking, and Random Forest implementation. | [View Notebook](./notebooks/chapter_07_ensemble_learning/chapter_07_notebook.ipynb) |
| 8 | Dimensionality Reduction | Techniques like PCA, Kernel PCA, and LLE. Focus on reducing complexity while preserving information. | [View Notebook](./notebooks/chapter_08_dimensionality_reduction/chapter_08_notebook.ipynb) |
| 9 | Unsupervised Learning Techniques | Covers clustering algorithms: K-Means, DBSCAN, Gaussian Mixture Models, and anomaly detection. | [View Notebook](./notebooks/chapter_09_unsupervised_learning/chapter_09_notebook.ipynb) |
| 10 | Introduction to Artificial Neural Networks | Fundamentals of ANNs: perceptrons, multilayer perceptrons, and training deep networks using backpropagation. | [View Notebook](./notebooks/chapter_10_introduction_to_anns/chapter_10_notebook.ipynb) |
| 11 | Training Deep Neural Networks | Deep learning optimization: initialization strategies, batch normalization, dropout, and adaptive optimizers. | [View Notebook](./notebooks/chapter_11_training_deep_neural_networks/chapter_11_notebook.ipynb) |
| 12 | Custom Models and Training with TensorFlow | Building custom training loops, using TensorFlow’s low-level API, and managing computational graphs. | [View Notebook](./notebooks/chapter_12_custom_models_tensorflow/chapter_12_notebook.ipynb) |
| 13 | Loading and Preprocessing Data with TensorFlow | Data ingestion pipelines using the `tf.data` API, preprocessing large datasets efficiently. | [View Notebook](./notebooks/chapter_13_data_preprocessing_tensorflow/chapter_13_notebook.ipynb) |
| 14 | Deep Computer Vision Using Convolutional Neural Networks | Introduction to CNNs, convolutional layers, pooling, padding, and architecture design for image classification. | [View Notebook](./notebooks/chapter_14_convolutional_neural_networks/chapter_14_notebook.ipynb) |
| 15 | Processing Sequences Using RNNs and CNNs | Covers sequential data handling: RNNs, LSTMs, GRUs, and 1D convolutions for text and time series. | [View Notebook](./notebooks/chapter_15_recurrent_neural_networks/chapter_15_notebook.ipynb) |
| 16 | Natural Language Processing with RNNs and Attention | NLP with RNNs, encoder-decoder models, and attention mechanisms for translation and text generation. | [View Notebook](./notebooks/chapter_16_nlp_attention/chapter_16_notebook.ipynb) |
| 17 | Representation Learning and Autoencoders | Autoencoders, variational autoencoders (VAEs), and generative models for dimensionality reduction and denoising. | [View Notebook](./notebooks/chapter_17_autoencoders/chapter_17_notebook.ipynb) |
| 18 | Generative Adversarial Networks (GANs) | Introduction to GANs: architecture, training stability issues, and creative applications in image generation. | [View Notebook](./notebooks/chapter_18_gans/chapter_18_notebook.ipynb) |
| 19 | Reinforcement Learning | RL fundamentals: agents, environments, rewards, and algorithms like Q-learning and policy gradients. | [View Notebook](./notebooks/chapter_19_reinforcement_learning/chapter_19_notebook.ipynb) |
| 20 | Training and Deploying TensorFlow Models at Scale | Model exporting, TensorFlow Serving, TensorFlow Lite, and deployment workflows for production. | [View Notebook](./notebooks/chapter_20_deployment/chapter_20_notebook.ipynb) |

---

### 🧠 Key Learning Goals
- Build, train, and evaluate ML models with Scikit-Learn  
- Design reproducible ML workflows  
- Understand feature engineering and data preprocessing  
- Explore deep learning and neural networks with TensorFlow/Keras  
- Deploy and scale models for real-world use  

---

### ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/umbutun/hands-on-machine-learning-practice.git
cd hands-on-machine-learning-practice

# Create environment (if using Conda)
conda env create -f environment.yml
conda activate hands-on-ml

# Or install dependencies via pip
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```
---
### 🧩 Dependencies
	•	Python 3.10+
	•	Scikit-Learn
	•	NumPy, Pandas, Matplotlib, Seaborn
	•	TensorFlow / Keras
	•	JupyterLab

(See requirements.txt for full list.)

---

### 🏆 Highlights
	•	✅ Modular structure by chapter
	•	✅ Real projects linked separately (e.g. Real Estate Price Predictor￼)
	•	✅ Reproducible environments
	•	✅ Learning-first, project-based approach

---

### 🧑‍💻 Author

• Umut Bütün
📍 Machine Learning Enthusiast | Python Developer
🔗 [GitHub Profile](https://github.com/umbutun)￼

---

### 📜 License

This repository is licensed under the MIT License￼.

---

### “Learning by doing — one model, one notebook at a time.”
