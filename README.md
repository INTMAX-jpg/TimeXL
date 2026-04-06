# TimeEncoder: Explainable Weather Forecasting with Prototype-based Encoder
(Based on the idea of the NeurIPS 2025 paper: *Explainable Multimodal Time Series Prediction with LLM-in-the-Loop*, hereinafter referred to as the original paper)

## Project Introduction
🌈 The original paper proposes the **TimeXL** framework, an explainable deep learning framework designed for time series prediction. It combines the efficiency of **prototype learning** with the reasoning capability of **Large Language Models (LLMs)**.

💪 Motivation: I am deeply impressed by the core idea of this paper, yet there is no official open-source implementation available. Therefore, in this semester's course project, I attempt to implement the TimeXL framework in engineering practice.

## Current Work: Encoder_Demo_for_HHWD
This repository has completed the preliminary implementation of the prototype-based encoder module. The encoder can learn typical weather patterns (namely *prototypes*) from historical meteorological data, and realize case-based reasoning for prediction explanation (e.g., *"I predict rainfall because the current weather pattern is highly similar to these 3 historical rainy days..."*).

The model is trained and evaluated on the public dataset **Historical Hourly Weather Data 2012-2017 (HHWD)** from Kaggle.

### Project Structure
```text
Basic_imple_of_Encoder_for_HHWD/   # Workable encoder implementation on HHWD dataset
├── data/                          # Scripts for data loading and preprocessing
│   ├── historical-hourly-weather-data/  # Raw CSV datasets
│   ├── processed_data/            # Processed PyTorch tensors (.pt)
│   ├── preprocess_data.py         # Script for data cleaning and splitting
│   └── real_data_loader.py        # PyTorch Dataset implementation
├── pth/                           # Saved model checkpoints
├── src/                           # Main executable scripts
│   ├── train_encoder.py           # Training script for the prototype-based encoder
│   ├── evaluate_encoder.py        # Model evaluation (Acc, KL, MAE)
│   └── interactive_predict.py     # Interactive CLI prediction tool
└── training/                      # Core model architecture, loss functions and trainers
    ├── models.py                  # Implementation of TimeXLModel and PrototypeManager
    ├── loss.py                    # Custom loss (KL Divergence + Prototype Losses)
    └── base_trainer.py            # Training loop and prototype projection for interpretability
```

### Prediction Task
⭐ Input: Meteorological description texts for **24 consecutive hours**
⭐ Output: Probability distribution of three weather categories (Rain, Snow, Other) in the next 24 hours

### Experimental Results
🤩 On the test set, the KL divergence between the predicted distribution and the real distribution is only **0.08**, achieving excellent performance.
It shows that the standalone encoder performs well for coarse-grained classification prediction on the HHWD dataset.

---

## Quick Start
You can quickly experience the project with the following commands:

### 1. Train the Encoder
Run the training script, and the model weights and prototypes will be saved to the `pth` directory automatically.
```bash
python Basic_imple_of_Encoder_for_HHWD\src\train_encoder.py
```

### 2. Model Evaluation
Evaluate model metrics (accuracy, KL divergence, MAE) on the test set
```bash
python Basic_imple_of_Encoder_for_HHWD\src\evaluate_encoder.py
```

### 3. Interactive Prediction
Launch the command-line interactive prediction tool for custom input testing
```bash
python Basic_imple_of_Encoder_for_HHWD\src\interactive_predict.py
```

#### Input Instructions
- **Required**: Input meteorological description texts of the past 24 hours
  ```
  sky is clear, scattered clouds, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist, mist
  ```
- **Optional**: Input time-series numerical data of the past 24 hours (all-zero vectors are used by default without input)

The program will output prediction results and explanations (text sequences corresponding to the best-matched prototypes).

---

## Future Work: TimeXL(not_finished)
In addition to the usable encoder demo above, I have implemented the core components of the complete TimeXL framework following the original paper for reference by subsequent developers.

```text
TimeXL_implementation(not_finished)/ # Work-in-progress full implementation of Algorithm 1 in the paper
├── Algorithm_1(not_finished).py   # Main iterative optimization loop
├── encoders.py                    # Time and text encoders
├── llm_agents.py                  # LLM interaction agents
├── losses.py                      # Loss functions
├── prototypes.py                  # Prototype management
└── trainer.py                     # Training loop
```