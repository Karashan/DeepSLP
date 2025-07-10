# DeepSLP

**DeepSLP: A Deep Neural Network Framework Leveraging Cancer Dependency Data to Predict Synthetic Lethality in Human Cells**

DeepSLP is a Python-based framework implementing a deep learning approach to predict synthetic lethal (SL) interactions from functional genomics features derived from the Cancer Dependency Map (DepMap). It provides tools to construct pairwise feature representations from CRISPR knockout fitness profiles and gene expression data, train deep neural network models, and benchmark prediction performance.

---

## ✨ Features

- **Pairwise Feature Engineering**  
  Construct embeddings from DepMap gene-effect and expression data for genome-wide coverage.

- **Deep Neural Network Model**  
  PyTorch implementation of scalable SL predictors with configurable architectures.

- **Benchmarking Utilities**  
  Evaluation scripts for in-context and cross-context generalization testing.

- **Reproducible Environment**  
  Conda YAML file provided to set up all dependencies easily.

---

## 📂 Project Structure

```
DeepSLP/
├── data/                # Input data files (DepMap profiles, labels, etc.)
├── models/              # Model architecture and training code
├── notebooks/           # Example Jupyter notebooks for analysis
├── scripts/             # CLI scripts for preprocessing, training, evaluation
├── utils/               # Helper functions and utilities
├── environment.yml      # Conda environment specification
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/csbio/DeepSLP.git
   cd DeepSLP
   ```

2. **Create the environment**

   ```bash
   conda env create -f environment.yml
   conda activate deepslp
   ```

3. **Run example training**

   ```bash
   python scripts/train_model.py --config configs/example_config.yaml
   ```

4. **Evaluate model performance**

   ```bash
   python scripts/evaluate_model.py --model checkpoints/best_model.pth
   ```

---

## 📖 Documentation

See the example Jupyter notebooks in `notebooks/` for walkthroughs on:
- Loading and preprocessing DepMap data
- Creating pairwise feature matrices
- Training and evaluating the model

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Citation

If you use DeepSLP in your research, please cite:

> [Zhang et al.], "DeepSLP: a deep neural network framework leveraging cancer dependency data to predict synthetic lethality in human cells," [Journal Name], [Year].

---

## ✉️ Contact

For questions or contributions, please open an issue or contact Xiang Zhang [zhangab18@gmail.com].