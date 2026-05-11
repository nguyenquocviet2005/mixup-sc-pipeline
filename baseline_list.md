# Baselines Used in Experiments

The following table lists the baselines and methods used in the selective classification and out-of-distribution detection experiments, along with their corresponding paper titles.

| Method Name | Paper Title |
| :--- | :--- |
| **Baseline Mixup** | *mixup: Beyond Empirical Risk Minimization* (ICLR 2018) / Uses Maximum Softmax Probability (MSP) from *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks* (ICLR 2017) |
| **DOCTOR-Alpha / DOCTOR-Beta** | *DOCTOR: A Simple Method for Detecting Misclassification Errors* (NeurIPS 2021) |
| **Energy Score** | *Energy-based Out-of-Distribution Detection* (NeurIPS 2020) |
| **KNN-OOD (k=50)** | *Out-of-Distribution Detection with Deep Nearest Neighbors* (ICML 2022) |
| **Mahalanobis Distance** | *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks* (NeurIPS 2018) |
| **MaxLogit pNorm / MaxLogit pNorm+** | *How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks* (arXiv:2305.14371) |
| **ODIN** | *Enhancing The Reliability of Out-of-Distribution Image Detection in Neural Networks* (ICLR 2018) |
| **RL_conf-M (Logit Margin) / RL_geo-M (Geometric Margin)** | *Selective Classification Under Distribution Shifts* (arXiv:2405.05160) |
| **SIRC (MSP + ViM-Res)** | *Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data* (NeurIPS 2022) |
| **SR_ent (Negative Entropy)** | Standard baseline using predictive entropy. (Commonly used in literature, e.g., discussed alongside MSP) |
| **ViM (Virtual Logit Matching)** | *ViM: Out-of-Distribution Detection with Virtual-logit Matching* (CVPR 2022) |

*Note: **Method 2: Feature-kNN / Logit Probability Blending** is listed in the results but appears to be the proposed/tested method rather than a standard baseline. If there is a specific paper associated with it, please let me know to add it.*
