# Hippo-UQ: Biologically-Inspired Uncertainty Quantification & OOD Detection

This repository contains a Bayesian probabilistic model simulating how the human hippocampus tracks uncertainty, detects context switches, and updates beliefs. It models where a learning system switches from being **error-driven** (learning from the environment) to **prediction-driven** (stop learning, start assuming), as observed in recent fMRI studies (e.g., Aitken & Kok, 2022).

### Why this matters for AI Alignment
As machine learning models are deployed in complex environments, monolithic confidence scores and opaque loss functions are insufficient for safe decision-making. We need systems that can "know what they don't know." By looking inside the "black box" of biological belief updating, this project explores concepts critical to AI alignment:

1. **Out-of-Distribution (OOD) Detection:** The model explicitly separates *irreducible noise* (aleatoric uncertainty) from *model ignorance/distribution shifts* (epistemic uncertainty). When the environment resets (a context switch), the model registers an OOD event, spiking epistemic uncertainty to rapidly unlearn old priors and safely adapt.
2. **Mechanistic Interpretability:** Instead of a single confidence output, the model disentangles Prediction ($P$), Precision/Stability ($I$), and Prediction Error ($PE$). This allows us to mechanistically track how the system's internal representations are weighted during distribution shifts.
3. **Hallucination vs. Perception:** The code models how high-precision priors cause the system to suppress unexpected sensory evidence (invalid trials) and project its own expectations—a biological analog to model "hallucination" under overconfidence.

## The Biological Concept: "The Hippocampal Switch"

In stable environments, the hippocampus uses learned rules to predict the future. However, under uncertainty or volatility, it must drop its predictions and focus heavily on immediate sensory errors to learn the new rules.

This code replicates the "Crossover Effect" of hippocampal representation:
* **Early Phase (High Uncertainty / OOD):** The model doesn't know the rule. Precision ($I$) is low. The system acts like a camera, recording exactly what happens (high Prediction Error weighting). Both valid and invalid data are encoded equally.
* **Late Phase (High Precision / In-Distribution):** The model has learned the rule. Precision ($I$) is high. The system acts like a projector. It strongly enhances expected data (Valid trials) and actively suppresses unexpected data (Invalid trials), driving the evidence for the presented shape below baseline.

## Repository Structure

* `optlearner.py`: The core Bayesian Probability Learner (adapted from Behrens et al., 2007). It maintains a joint probability grid of predictions ($P$) and inertias/volatility ($I$), updating beliefs via exact Bayesian inference.
* `simulate_data.py`: Generates environments with hidden probabilistic rules (e.g., 75% predictive accuracy) and periodic forced "context switches" (OOD events) to test the model's adaptation. It tracks metrics like $PE \times U$ (Prediction Error $\times$ Uncertainty) and centered $P \times I$ (Prediction $\times$ Precision).
* `data_processing.py`: Handles the complex logic of pooling trial data across multiple blocks, properly aligning "Valid" (expected) and "Invalid" (unexpected) trials without leaking future data during the critical context-switch boundaries.
* `plotting.py`: Visualizes the internal state of the model. It generates smoothed, publication-ready plots demonstrating the crossover between error-driven and prediction-driven representations.

## The Math: Isolating the Representations

To effectively peek inside the black box and distinguish between what the model *expects* versus what it *observes*, the code calculates the **Evidence for the Presented Shape**. 

We center the probability to a baseline of 0 (where chance = 0.5) and scale it by the model's precision ($I$). 
* **Valid Trial:** The model predicts Shape A, and sees Shape A. 
    `(+ centered_p) * I` $\rightarrow$ Massive Positive Evidence.
* **Invalid Trial:** The model confidently predicts Shape A, but sees Shape B.
    `(- centered_p) * I` $\rightarrow$ Massive Negative/Suppressed Evidence.

This mathematical construction flawlessly mimics fMRI decoding techniques, revealing how high-confidence priors override sensory input.

## Usage

To run the simulation and generate the visualizations:

1. Ensure you have the required dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, and `scikit-learn`.
2. Run the plotting script:
```bash
python plotting.py
