# Mathematical Derivations: Range Estimation, CRLB, & Weight Update Dynamics

This document provides the formal mathematical derivations for the monocular drone detection, ranging, and kinematics engine implemented in the system.

---

## 1. Derivation 1: Geometric Perspective Projection & Sensitivity Analysis

### 1.1 Forward Observation Model (Pinhole Camera)
Under the standard pinhole camera projection model with focal length $f$ (in pixels), an object of known physical dimension $W$ (in meters) located at distance $D$ (in meters) projects onto the image sensor plane with a pixel width $w_{\text{px}}$:

$$w_{\text{px}} = \frac{W \cdot f}{D}$$

Similarly, for physical height $H$ and diagonal $L = \sqrt{W^2 + H^2}$:

$$h_{\text{px}} = \frac{H \cdot f}{D}, \qquad d_{\text{px}} = \frac{\sqrt{W^2 + H^2} \cdot f}{D}$$

### 1.2 Deterministic Monocular Distance Inversion
Rearranging the forward projection gives the deterministic distance estimator:

$$D = \frac{W \cdot f}{w_{\text{px}}}$$

### 1.3 Sensitivity Derivative & First-Order Error Propagation
To determine how bounding box localization errors $\Delta w_{\text{px}}$ propagate into distance uncertainty $\Delta D$, we compute the first-order partial derivative:

$$\frac{\partial D}{\partial w_{\text{px}}} = \frac{d}{d w_{\text{px}}} \left( W f w_{\text{px}}^{-1} \right) = -W f w_{\text{px}}^{-2} = -\frac{W \cdot f}{w_{\text{px}}^2}$$

Substituting $w_{\text{px}} = \frac{W f}{D}$:

$$\frac{\partial D}{\partial w_{\text{px}}} = -\frac{W \cdot f}{\left(\frac{W f}{D}\right)^2} = -\frac{D^2}{W \cdot f}$$

**Physical Implication:** Distance error grows quadratically ($\mathcal{O}(D^2)$) with target range. At 30 meters, a 1-pixel regression error induces 100$\times$ greater distance error than at 3 meters.

---

## 2. Derivation 2: Fisher Information Matrix (FIM) & Cramér-Rao Lower Bound (CRLB)

### 2.1 Statistical Measurement Model
Let the observed bounding box width $w$ be corrupted by zero-mean additive Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma_w^2)$ representing YOLO boundary regression uncertainty:

$$w = g(D) + \epsilon = \frac{W \cdot f}{D} + \epsilon$$

The likelihood function $p(w | D)$ is:

$$p(w | D) = \frac{1}{\sqrt{2\pi\sigma_w^2}} \exp\left( -\frac{\left(w - \frac{W f}{D}\right)^2}{2\sigma_w^2} \right)$$

### 2.2 Log-Likelihood & Score Function
Taking the natural logarithm of the likelihood:

$$\ln p(w | D) = -\frac{1}{2} \ln(2\pi\sigma_w^2) - \frac{1}{2\sigma_w^2} \left( w - \frac{W f}{D} \right)^2$$

Differentiating with respect to the parameter of interest $D$ gives the score function $S(D)$:

$$S(D) = \frac{\partial \ln p(w | D)}{\partial D} = -\frac{1}{2\sigma_w^2} \cdot 2 \left( w - \frac{W f}{D} \right) \cdot \left( \frac{W f}{D^2} \right) = \frac{W f}{\sigma_w^2 D^2} \left( w - \frac{W f}{D} \right)$$

### 2.3 Fisher Information $I(D)$
The Fisher Information is defined as the expected value of the squared score function:

$$I(D) = \mathbb{E}\left[ \left(\frac{\partial \ln p(w | D)}{\partial D}\right)^2 \right] = \mathbb{E}\left[ \frac{W^2 f^2}{\sigma_w^4 D^4} \left( w - \frac{W f}{D} \right)^2 \right]$$

Since $\mathbb{E}\left[\left(w - \frac{W f}{D}\right)^2\right] = \text{Var}(\epsilon) = \sigma_w^2$:

$$I(D) = \frac{W^2 f^2}{\sigma_w^4 D^4} \cdot \sigma_w^2 = \frac{W^2 f^2}{\sigma_w^2 D^4}$$

### 2.4 Cramér-Rao Lower Bound
The Cramér-Rao inequality states that the variance of any unbiased estimator $\hat{D}$ is lower-bounded by the inverse of the Fisher Information:

$$\text{Var}(\hat{D}) \geq \text{CRLB}(D) = \frac{1}{I(D)} = \frac{\sigma_w^2 D^4}{W^2 f^2}$$

### 2.5 Total Theoretical Standard Deviation
Incorporating out-of-plane 3D aspect ratio / rotation uncertainty ($\sigma_{\text{pose}}$):

$$\sigma_D = \sqrt{\text{CRLB}_{\text{pixel}}(D) + \sigma_{\text{pose}}^2(D)} = \sqrt{\frac{\sigma_w^2 D^4}{W^2 f^2} + (\kappa_{\text{pose}} \cdot D)^2}$$

---

## 3. Derivation 3: Mathematical Derivations for How Weights Were Changed

This section provides the rigorous derivations for the three weighting mechanisms operating in the system:
1. **Part 3A:** Optimal Inverse-Variance Fusion Weights (Multi-Cue Range Combination).
2. **Part 3B:** Dynamic Kalman Filter Gain Weights (Temporal Distance Filtering).
3. **Part 3C:** Neural Network Loss Gradient Weights (Bounding Box Regression & Classifier Updates).

---

### Part 3A: Optimal Inverse-Variance Fusion Weights (Best Linear Unbiased Estimator - BLUE)

#### Problem Statement
Given $M$ independent, unbiased range estimates $D_1, D_2, \dots, D_M$ derived from different geometric cues (width, height, diagonal) with individual CRLB variances $\sigma_1^2, \sigma_2^2, \dots, \sigma_M^2$, find the optimal linear combination:

$$D^* = \sum_{i=1}^M w_i D_i$$

subject to the unbiasedness constraint $\sum_{i=1}^M w_i = 1$, such that the total estimation variance $\text{Var}(D^*)$ is strictly minimized.

#### Objective Function & Lagrange Multiplier
The variance of the linear combination of independent estimators is:

$$\text{Var}(D^*) = \sum_{i=1}^M w_i^2 \sigma_i^2$$

Formulate the Lagrangian $\mathcal{L}(w_1, \dots, w_M, \lambda)$ with constraint $\sum_{i=1}^M w_i - 1 = 0$:

$$\mathcal{L}(w_1, \dots, w_M, \lambda) = \sum_{i=1}^M w_i^2 \sigma_i^2 - \lambda \left( \sum_{i=1}^M w_i - 1 \right)$$

#### Partial Derivatives & Optimality Condition
Taking the partial derivative with respect to each weight $w_k$:

$$\frac{\partial \mathcal{L}}{\partial w_k} = 2 w_k \sigma_k^2 - \lambda = 0 \implies w_k = \frac{\lambda}{2 \sigma_k^2}$$

Taking the partial derivative with respect to $\lambda$:

$$\frac{\partial \mathcal{L}}{\partial \lambda} = \sum_{i=1}^M w_i - 1 = 0 \implies \sum_{i=1}^M \frac{\lambda}{2 \sigma_i^2} = 1$$

Solving for $\lambda$:

$$\frac{\lambda}{2} \sum_{i=1}^M \frac{1}{\sigma_i^2} = 1 \implies \frac{\lambda}{2} = \frac{1}{\sum_{i=1}^M \frac{1}{\sigma_i^2}}$$

#### Final Optimal Weight Formula
Substituting $\frac{\lambda}{2}$ back into the expression for $w_k$:

$$w_k^* = \frac{\frac{1}{\sigma_k^2}}{\sum_{j=1}^M \frac{1}{\sigma_j^2}}$$

#### Minimum Resulting Variance
The minimum achievable variance of the fused estimator is:

$$\text{Var}(D^*) = \sum_{k=1}^M \left( \frac{\frac{1}{\sigma_k^2}}{\sum_{j=1}^M \frac{1}{\sigma_j^2}} \right)^2 \sigma_k^2 = \frac{\sum_{k=1}^M \frac{1}{\sigma_k^2}}{\left(\sum_{j=1}^M \frac{1}{\sigma_j^2}\right)^2} = \frac{1}{\sum_{j=1}^M \frac{1}{\sigma_j^2}} = \left( \sum_{j=1}^M \text{CRLB}(D_j)^{-1} \right)^{-1}$$

**Significance:** Each cue is weighted strictly proportionally to its Fisher Information $I(D_k) = \frac{1}{\sigma_k^2}$. Dimensions with low pixel uncertainty or large physical baseline dominate the fusion, while distorted dimensions are suppressed.

---

### Part 3B: Dynamic Kalman Filter Gain Weight Derivation

#### State Representation & Kinematics
Let the 1D range state vector at frame $k$ be $\mathbf{x}_k = \begin{bmatrix} z_k \\ \dot{z}_k \end{bmatrix}$ (distance and closing velocity):

$$\mathbf{x}_k = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{w}_k, \qquad \mathbf{F} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$$

Measurement model:

$$z_k = \mathbf{H} \mathbf{x}_k + v_k, \qquad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

where $v_k \sim \mathcal{N}(0, R_k)$ and the measurement noise covariance is dynamically set to the CRLB variance:

$$R_k = \sigma_{\text{CRLB}}^2(z_k)$$

#### A Priori Prediction
$$\hat{\mathbf{x}}_k^- = \mathbf{F} \hat{\mathbf{x}}_{k-1}$$
$$\mathbf{P}_k^- = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$$

#### A Posteriori State Weighting (Kalman Gain)
The updated state estimate is a weighted combination of prediction $\hat{\mathbf{x}}_k^-$ and new measurement $z_k$:

$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \left( z_k - \mathbf{H} \hat{\mathbf{x}}_k^- \right) = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \hat{\mathbf{x}}_k^- + \mathbf{K}_k z_k$$

To minimize the trace of the error covariance $\mathbf{P}_k = \mathbb{E}[(\mathbf{x}_k - \hat{\mathbf{x}}_k)(\mathbf{x}_k - \hat{\mathbf{x}}_k)^T]$:

$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k R_k \mathbf{K}_k^T$$

Differentiating $\text{Tr}(\mathbf{P}_k)$ with respect to $\mathbf{K}_k$ and setting to zero:

$$\frac{\partial \text{Tr}(\mathbf{P}_k)}{\partial \mathbf{K}_k} = -2 (\mathbf{P}_k^- \mathbf{H}^T) + 2 \mathbf{K}_k (\mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + R_k) = 0$$

Solving for the optimal Kalman weight matrix $\mathbf{K}_k$:

$$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T \left( \mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + R_k \right)^{-1}$$

**Dynamic Weight Adaptation:**
- At **close range** ($D < 5\text{m}$), $R_k \to 0 \implies \mathbf{K}_k \to \mathbf{H}^{-1} = [1, \frac{1}{\Delta t}]^T$. Measurement weight is maximized for instantaneous tracking.
- At **long range** ($D > 25\text{m}$), $R_k \propto D^4 \to \text{large} \implies \mathbf{K}_k \to 0$. State relies on kinematic momentum prediction, filtering out noisy pixel oscillations.

---

### Part 3C: Neural Network Backpropagation Weight Update Derivation (YOLO Box Regression & Classifier)

#### 1. Complete Loss Function
The total multi-task detection loss is composed of box regression loss $\mathcal{L}_{\text{box}}$ (CIoU / Complete IoU), distribution focal loss $\mathcal{L}_{\text{dfl}}$, and classification loss $\mathcal{L}_{\text{cls}}$:

$$\mathcal{L}_{\text{total}}(\mathbf{\Theta}) = \lambda_{\text{box}} \mathcal{L}_{\text{box}}(\mathbf{\Theta}) + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}(\mathbf{\Theta}) + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}(\mathbf{\Theta})$$

where $\mathbf{\Theta} = \{W^{(l)}, b^{(l)}\}_{l=1}^L$ represents all trainable neural network layer weights.

#### 2. Complete IoU (CIoU) Loss Formulation
For predicted box $B = (x_c, y_c, w, h)$ and ground truth $B^{gt} = (x_c^{gt}, y_c^{gt}, w^{gt}, h^{gt})$:

$$\mathcal{L}_{\text{box}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

where:
- $\rho(b, b^{gt})$ is the Euclidean distance between center points: $\rho^2 = (x_c - x_c^{gt})^2 + (y_c - y_c^{gt})^2$
- $c$ is the diagonal length of the smallest enclosing bounding box
- $v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2$ measures aspect ratio consistency
- $\alpha = \frac{v}{(1 - \text{IoU}) + v}$ is the adaptive aspect ratio weighting factor

#### 3. Gradient Computation & Backpropagation Chain Rule
For any neural network weight $W_{ij}^{(l)}$ connecting neuron $j$ in layer $l-1$ to neuron $i$ in layer $l$:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}^{(l)}} = \frac{\partial \mathcal{L}_{\text{total}}}{\partial z_i^{(l)}} \cdot \frac{\partial z_i^{(l)}}{\partial W_{ij}^{(l)}}$$

where $z_i^{(l)} = \sum_{k} W_{ik}^{(l)} a_k^{(l-1)} + b_i^{(l)}$. Since $\frac{\partial z_i^{(l)}}{\partial W_{ij}^{(l)}} = a_j^{(l-1)}$:

$$\delta_i^{(l)} \equiv \frac{\partial \mathcal{L}_{\text{total}}}{\partial z_i^{(l)}}$$
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}^{(l)}} = \delta_i^{(l)} a_j^{(l-1)}$$

For hidden layers $l$, $\delta^{(l)}$ is computed recursively via backpropagation:

$$\delta_i^{(l)} = \left( \sum_{k} \delta_k^{(l+1)} W_{ki}^{(l+1)} \right) \cdot \sigma'\left( z_i^{(l)} \right)$$

#### 4. Stochastic Gradient Descent with Momentum Weight Update Rule
At training step $t$, the weight update with learning rate $\eta$, momentum $\beta$, and weight decay $\mu$ is:

$$V_{ij}^{(l)}(t) = \beta V_{ij}^{(l)}(t-1) + \eta \left( \frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}^{(l)}} + \mu W_{ij}^{(l)}(t-1) \right)$$

$$W_{ij}^{(l)}(t) = W_{ij}^{(l)}(t-1) - V_{ij}^{(l)}(t)$$

**Summary Table of Weight Mechanisms:**

| Mechanism | Weight Symbol | Update / Governing Law | Role in Drone System |
| :--- | :--- | :--- | :--- |
| **Multi-Cue Fusion** | $w_k^*$ | $w_k^* = \frac{1/\sigma_k^2}{\sum_j 1/\sigma_j^2}$ | Fuses width, height, and diagonal distance estimates via Fisher Information |
| **Kalman Filter** | $\mathbf{K}_k$ | $\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T (\mathbf{H}\mathbf{P}_k^-\mathbf{H}^T + R_k)^{-1}$ | Dynamically balances kinematic model vs CRLB measurement noise |
| **YOLO Neural Network** | $\Delta W_{ij}$ | $\Delta W_{ij} = -\eta \frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}} + \beta V_{ij}(t-1)$ | Updates convolutional kernels and box regression heads via backprop |
