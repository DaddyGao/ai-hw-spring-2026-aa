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

# Demo

## 1. Introduction & Objective
* **Goal:** Evaluate the adversarial robustness of a standard Deep Learning classifier.
* **Target Dataset:** MNIST handwritten digits (60,000 train, 10,000 test images).
* **Core Task:** 1. Train a Convolutional Neural Network (CNN) to high accuracy.
  2. Implement three white-box adversarial attacks to intentionally fool it.
  3. Compare attack effectiveness using **Attack Success Rate (ASR)**.

*🗣️ **Notes:** The core objective here is to explore how fragile modern neural networks are. Even when a model is highly accurate under normal conditions, it can be completely blinded by mathematically optimized noise. I evaluated three prominent white-box attacks: FGSM, PGD, and Momentum-based iterative FGSM.*

---

## 2. Target Model & Training Progress
* **Architecture:** 2-Layer Convolutional Neural Network (CNN) with Dropout and Max Pooling.
* **Training Duration:** 3 Epochs.
* **Loss Convergence:**
  * **Epoch 1 End:** Loss dropped from 2.31 to 0.21
  * **Epoch 2 End:** Loss minimized to 0.077
  * **Epoch 3 End:** Final training loss converged to **0.0078**

*🗣️ **Notes:** Before attacking a model, we first need a strong model to serve as our target. I built a standard CNN. Over 3 epochs, the training loss dropped significantly from a random guessing loss of 2.31 all the way down to a near-zero loss of 0.0078. This indicates the network successfully converged and mastered the training data.*

---

## 3. Baseline & Attack Setup
* **Baseline Recognition Rate:** **98.77%** Clean Accuracy on the test set.
* **Perturbation Budget (Epsilon $\epsilon$):** Set to **0.3**.
  * This restricts the maximum modification allowed per pixel to 30%.
* **The Metric - Attack Success Rate (ASR):**
  $$\text{ASR} = \frac{\text{Successfully Fooled Clean Images}}{\text{Total Originally Correct Clean Images}} \times 100$$
  * *Crucial Rule:* ASR only measures performance on the 98.77% of images the model actually got right initially.

*🗣️ **Notes:** When evaluated on the clean, unattacked test dataset, our model achieved a recognition rate of 98.77%. For our attacks, we set the perturbation constraint, or Epsilon, to 0.3. This means we allow the attack algorithms to shift any pixel value by up to 30%. To track performance, we use Attack Success Rate. It's important to note that ASR is only calculated using the images the model originally classified correctly, as you can't logically 'fool' a model into getting an answer wrong if it was already going to fail that image naturally.*

---

## 4. Experimental Results
Here is the performance breakdown of the model under an adversarial environment:

| Evaluation Scenario | Epsilon ($\epsilon$) | Performance Metric | Value |
| :--- | :---: | :---: | :---: |
| **Clean Baseline** | 0.0 | Recognition Rate | **98.77%** |
| **FGSM Attack** | 0.3 | Attack Success Rate | **94.21%** |
| **PGD Attack** | 0.3 | Attack Success Rate | **100.00%** |
| **MI-FGSM Attack** | 0.3 | Attack Success Rate | **100.00%** |

*🗣️ **Notes:** Here are the concrete findings from the experiment. While the baseline accuracy sits proudly close to 99%, the adversarial attacks completely dismantled it. The single-step FGSM achieved a massive 94.21% success rate. Even more severely, both the iterative attacks—PGD and Momentum-based FGSM—achieved a perfect 100% Attack Success Rate. This means they wiped out the model's accuracy completely.*

---

## 5. Single-Step vs. Iterative Attacks
* **FGSM (Fast Gradient Sign Method): 94.21% ASR**
  * Calculates the gradient of the loss once, then takes a single large step in that direction.
  * Fast, but can over-approximate and leave a small margin of surviving correct digits.
* **PGD & MI-FGSM: 100.00% ASR**
  * PGD takes 40 small steps instead of one large step, continuously projecting back into the $\epsilon$-ball.
  * MI-FGSM adds a momentum term to bypass local extrema in the loss landscape.
  * These multi-step methods find the absolute optimal adversarial noise, causing total model failure.

*🗣️ **Notes:** Why is there a gap between 94% and 100%? It comes down to optimization. FGSM is a single-step approach; it looks at the gradient once and leaps in the direction of maximum loss. It's fast, but crude. PGD and MI-FGSM, on the other hand, are iterative. In this script, they took 40 tiny steps, recalculating the gradient at every point, and ensuring they stayed inside our 0.3 boundary. MI-FGSM adds a momentum buffer so it doesn't get stuck in mathematical ruts. This fine-grained tuning allows them to perfectly exploit the decision boundaries of the model, resulting in a flawless 100% exploit rate.*

---

## 6. Conclusion
* **Key Takeaway:** High baseline accuracy does **not** equal structural model security.
* Standard empirical risk minimization makes models highly vulnerable to intentional out-of-distribution math noise.
* **Future Work:** To mitigate this absolute failure, the network would require defensive strategies like **Adversarial Training** (injecting PGD examples back into the training loop).
* Thank you! Any questions?

*🗣️ **Notes:** In conclusion, this experiment teaches us a vital lesson in machine learning security: high accuracy under normal testing criteria does not mean a model is robust. Standard networks are highly fragile to worst-case distribution shifts. To fix this 100% failure rate, future iterations of this model would require defense mechanisms like adversarial training—where the model trains directly on PGD examples to learn how to resist them. Thank you for your time, and I'd be happy to take any questions.*
