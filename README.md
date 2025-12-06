# CSE473s: Build Your Own Neural Network Library

**Course:** CSE473s - Computational Intelligence (Fall 2025)  
**Institution:** Faculty of Engineering, Ain Shams University  

## 📌 Project Overview
This project involves developing a foundational neural network library from scratch using **only Python and NumPy**. The goal is to demystify deep learning frameworks by implementing the core mathematics of forward propagation, backpropagation, and optimization manually.

The library is validated through three major tasks:
1.  **Solving the XOR Problem:** Proving the library can learn non-linear decision boundaries.
2.  **Unsupervised Learning (Autoencoder):** Compressing and reconstructing MNIST images.
3.  **Latent Space Classification:** Using the trained encoder as a feature extractor for an SVM classifier.

---

## 📂 Repository Structure
The project follows a modular architecture as required:

```text
├── lib/                    # The Core Neural Network Library
│   ├── __init__.py
│   ├── layers.py           # Dense layer implementation (Forward/Backward)
│   ├── activations.py      # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py           # Mean Squared Error (MSE)
│   ├── optimizer.py        # Stochastic Gradient Descent (SGD)
│   └── network.py          # Network class to orchestrate training
│
├── notebooks/              # Demos and Experiments
│   └── project_demo.ipynb  # Main demo (Gradient Check, XOR, Autoencoder, TF Comparison)
│
├── report/                 # Documentation
│   └── project report.pdf  # Final detailed report and analysis
│
├── .gitignore
├── README.md
└── requirements.txt        # Dependencies (NumPy, Matplotlib, TensorFlow)
````

-----

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
cd [CI_Project]
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

## 📊 Results Snapshot

### XOR Problem

  * **Converged Loss:** \~0.0001 (MSE)
  * **Accuracy:** 100%

### TensorFlow Benchmark

  * **Performance:** Our custom NumPy implementation proved significantly faster for small-scale datasets (XOR) compared to TensorFlow's computational graph overhead.
  * **Convergence:** Both models converged to near-zero loss, validating the correctness of our mathematical implementation.

-----

## 👥 Contributors

  * **Member 1:** [Adham Ehab Saleh]              - [2100679]
  * **Member 2:** [Ahmed Mohamed Ramadan]         - [2100323]
  * **Member 3:** [Ahmed Salah Eldin Abdelrahman] - [2100505]
  * **Member 4:** [Ahmed Yasser Hosney]           - [2101101]
  * **Member 5:** [Omar Magdy Abdelsattar]        - [2100273]

-----

*This project was submitted for the CSE473s Computational Intelligence course at Ain Shams University.*

```
