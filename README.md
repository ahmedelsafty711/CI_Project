# 🧠 Custom Neural Network Library
### CSE473s — Computational Intelligence Project


A foundational deep learning framework built from scratch using **only Python and NumPy**. This project demystifies modern deep learning libraries (like TensorFlow or PyTorch) by manually implementing the core mathematics of forward propagation, backpropagation, and optimization.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Features implemented](#-features-implemented) 
- [Installation & Usage](#-installation--usage)
- [Benchmarks & Results](#-benchmarks--results)
- [Contributors](#-contributors)

---

## 🔭 Overview
Developed for the **Fall 2025 CSE473s** course at **Ain Shams University**, this library serves as an educational tool to understand the inner workings of neural networks.

The library is robust enough to solve non-linear problems (like XOR), perform unsupervised learning (Autoencoders on MNIST), and act as a feature extractor for hybrid models.

---


## 🚀 Quick Start

Here is a simple example of how to use the library to create a network for the XOR problem:

```python
import numpy as np
from lib.network import Network
from lib.layers import Dense
from lib.activations import ReLU, Sigmoid

# 1. Prepare Data (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 2. Build the Model
model = Network()
model.add(Dense(input_size=2, output_size=4))  # Hidden Layer
model.add(ReLU())                              # Activation
model.add(Dense(input_size=4, output_size=1))  # Output Layer
model.add(Sigmoid())                           # Activation

# 3. Train
# Note: Adjust parameters (epochs, lr) based on your specific implementation
model.train(X, y, epochs=1000, learning_rate=0.1)

# 4. Predict
predictions = model.predict(X)
print("Predictions:", predictions)
```
----

## 📂 Repository Structure
The project follows a modular architecture as required:

```bash
CI_Project/
├── lib/                  # 📦 The Core Library
│   ├── activations.py    #    Activation classes (ReLU, Sigmoid, etc.)
│   ├── layers.py         #    Dense layer implementation
│   ├── losses.py         #    Loss functions (MSE)
│   ├── optimizer.py      #    SGD implementation
│   └── network.py        #    Network class (Trainer/Predictor)
│
├── notebooks/            # 📓 Experiments & Demos
│   └── project_demo.ipynb#    XOR, Autoencoder, and TF Benchmarks
│
├── report/               # 📄 Documentation
│   └── project report.pdf
│
├── requirements.txt      # 🧱 Dependencies
└── README.md             # 🏠 You are here
````

----

## 🚀 Features implemented

### 1\. Core Library (`lib/`)

  * **Layer Abstraction:** Base `Layer` class with extensible forward/backward methods.
  * **Dense Layer:** Fully connected layer supporting weights and biases gradients.
  * **Activations:** `Sigmoid`, `Tanh`, `ReLU`, `Softmax`.
  * **Loss Function:** Mean Squared Error (MSE) and its derivative.
  * **Optimizer:** Stochastic Gradient Descent (SGD).

### 2\. Validation & Testing (`notebooks/`)

  * **Gradient Checking:** Numerical verification of analytical gradients (Finite Difference Method).
  * **XOR Solution:** A 2-4-1 network solving the classic XOR problem with 100% accuracy.
  * **Benchmarking:** Performance comparison (Time & Loss) against **TensorFlow/Keras**.

-----

## 🛠️ Installation & Usage

### 1\. Clone the Repository

```bash
git clone https://github.com/ahmedelsafty711/CI_Project.git
cd CI_Project
```

### 2\. Set Up Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Run the Demo

Launch Jupyter Notebook to view the training demos and results:

```bash
jupyter notebook notebooks/project_demo.ipynb
```

-----

## 📊 Benchmarks & Results

### XOR Problem
* **Converged Loss:** ~0.0001 (MSE)
* **Accuracy:** 100%

### MNIST Autoencoder
* **Configuration:** 784-64-784
* **Accuracy/Loss:** Low Recon Loss
* **Notes:** Successful image reconstruction

### Comparisons
* **Configuration:** vs TensorFlow
* **Accuracy/Loss:** Faster
* **Notes:** Outperformed TF on small datasets

---

## 👥 Contributors

| Name | ID
| :--- | :--- 
| **Adham Ehab Saleh** | 2100679
| **Ahmed Mohamed Ramadan** | 2100323
| **Ahmed Salah Eldeen** | 2100505
| **Ahmed Yasser Hosney** | 2101101
| **Omar Magdy Abdelsattar**| 2100273

---

*This project was submitted for the CSE473s Computational Intelligence course at Ain Shams University.*


