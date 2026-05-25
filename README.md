# Attack MNIST Recognition

## Overview
This document contains the implementation testing results for adversarial attacks on a Convolutional Neural Network (CNN) trained on the MNIST dataset. The evaluated attacks are the Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and Momentum Iterative FGSM (MI-FGSM).

## Training Progress
The target model was trained for 3 epochs on the MNIST training dataset. The training loss demonstrated successful convergence:
* **Epoch 1:** Loss decreased from 2.309421 to 0.211362
* **Epoch 2:** Loss decreased from 0.079949 to 0.077817
* **Epoch 3:** Loss decreased from 0.063038 to a final loss of 0.007821

## Testing Results
After training, the target model was evaluated on the MNIST test set. Adversarial examples were generated using an **Epsilon ($\epsilon$) noise budget of 0.3**. 

### Baseline Performance
* **Recognition Rate (Clean Accuracy):** 98.77%
    * *The high clean accuracy indicates the baseline model learned the dataset effectively before being subjected to attacks.*

### Attack Success Rate (ASR)
*Note: ASR is measured as the percentage of successfully perturbed images, counted exclusively among the images the model originally classified correctly.*

| Attack Algorithm | Epsilon | Attack Success Rate | Description |
| :--- | :---: | :---: | :--- |
| **FGSM** | 0.3 | **94.21%** | Fast Gradient Sign Method (Single-step attack) |
| **PGD** | 0.3 | **100.00%** | Projected Gradient Descent (Iterative attack) |
| **MI-FGSM** | 0.3 | **100.00%** | Momentum Iterative FGSM |

## Analysis & Conclusion
* **Vulnerability to Single-Step Attacks:** FGSM proved highly effective with a 94.21% ASR. Taking a single step in the direction of the gradient using an $\epsilon$ of 0.3 is enough to drastically alter the model's predictions.
* **Total Devastation by Iterative Attacks:** Both PGD and MI-FGSM achieved a perfect **100.00% ASR**. By taking multiple smaller steps within the $\epsilon$-ball limits, these algorithms successfully found the precise pixel alterations needed to force a misclassification on *every single image* the model originally predicted correctly. This highlights the severe fragility of undefended neural networks against optimized adversarial perturbations.
