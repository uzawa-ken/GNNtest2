# PUP-HAW-U: Physics-based Uncertainty Propagation with Hierarchical Adaptive Weighting for Unsupervised PDE Solving

## 論文アウトライン / Paper Outline

---

## Abstract

**背景**: Graph Neural Networks (GNNs) have shown promise for PDE solving, but existing methods struggle with mesh quality variations and weight balancing in unsupervised learning.

**提案手法**: We propose PUP-HAW-U, a novel framework that combines:
1. Physics-based uncertainty propagation weighted by mesh quality and solution curvature
2. Topology-aware weight propagation for spatial influence modeling
3. Hierarchical adaptive weighting across multiple time scales
4. Multi-physics constraint balancing for fully unsupervised learning

**結果**: Experiments on CFD benchmarks demonstrate that PUP-HAW-U achieves X% improvement over fixed-weight baselines on low-quality meshes while maintaining competitive performance on high-quality meshes.

**インパクト**: Our approach enables accurate PDE solving without ground truth data, making it applicable to real-world scenarios where labeled data is scarce or expensive.

---

## 1. Introduction

### 1.1 Motivation

- Challenge: PDE solving requires accurate numerical solutions on complex geometries
- Traditional methods: CFD solvers are computationally expensive
- ML approach: GNNs can learn to solve PDEs faster
- **Key problem**: Existing methods require:
  - Large amounts of labeled data (expensive to generate)
  - Struggle with low-quality meshes
  - Manual weight tuning for multi-objective losses

### 1.2 Our Contribution

We propose PUP-HAW-U, addressing three key challenges:

1. **Physics-informed weighting** (Phase 2):
   - Mesh quality metrics + solution curvature
   - Automatic error amplification detection

2. **Hierarchical adaptation** (Phase 3):
   - Level 0: Data-adaptive reference values
   - Level 1: Epoch-wise global balancing
   - Level 2: Batch-wise gradient harmonization

3. **Fully unsupervised learning** (Phase 4-6):
   - PDE residual + BC + IC + conservation laws
   - No ground truth required for training
   - Automatic constraint balancing

### 1.3 Paper Organization

- Section 2: Related Work
- Section 3: Methodology (PUP-HAW-U)
- Section 4: Experimental Setup
- Section 5: Results and Analysis
- Section 6: Ablation Study
- Section 7: Conclusion

---

## 2. Related Work

### 2.1 Physics-Informed Neural Networks (PINNs)

- Original PINNs [Raissi et al., 2019]
- Limitations: Fixed loss weights, poor scaling

### 2.2 Graph Neural Networks for PDEs

- Message passing on meshes [Pfaff et al., 2021]
- MeshGraphNets [Sanchez-Gonzalez et al., 2020]
- Limitations: Supervised learning, uniform weighting

### 2.3 Multi-objective Loss Balancing

- Gradient normalization [Chen et al., 2018]
- Uncertainty weighting [Kendall et al., 2018]
- Learning by balancing (lbPINNs) [Wang et al., 2021]
- **Gap**: None specifically address mesh quality variations

### 2.4 Unsupervised PDE Learning

- Physics-informed approaches
- Boundary condition enforcement
- **Gap**: Limited success on complex geometries with poor mesh quality

---

## 3. Methodology: PUP-HAW-U

### 3.1 Problem Formulation

**Notation**:
- Graph: G = (V, E), |V| = N nodes, |E| = M edges
- PDE: Au = b (discretized system)
- Mesh quality: skewness, non-orthogonality, aspect ratio, size jump

**Objective**: Learn f_θ: X → u without ground truth u*

### 3.2 Phase 1: Baseline Architecture

- SimpleSAGE: 4-layer GraphSAGE
- Input: 13 features (coords, pressure, mesh quality metrics)
- Output: Pressure field

### 3.3 Phase 2: Physics-based Uncertainty Propagation

#### 3.3.1 Mesh Quality Factor

$$
\alpha(x_i) = \sum_{k=1}^{4} \beta_k \cdot \text{softplus}\left(\frac{q_k(x_i) - q_k^{\text{ref}}}{\sigma_k}\right)
$$

where:
- q_k: mesh quality metrics (skew, non-orth, AR, size jump)
- β_k: learnable coefficients
- q^ref, σ: reference values and scales

#### 3.3.2 Solution Curvature

$$
\kappa(x_i) = |\nabla^2 \phi(x_i)| \approx \left|\sum_{j \in \mathcal{N}(i)} L_{ij} \phi_j\right|
$$

Computed via graph Laplacian approximation

#### 3.3.3 Uncertainty Weight

$$
w_i = 1 + \alpha(x_i) \cdot (1 + \kappa(x_i))
$$

#### 3.3.4 Topology-aware Propagation

$$
\tilde{w}_i = \max_{h=0}^{H} \left\{ \gamma^h \cdot \max_{j \in \mathcal{N}^h(i)} w_j \right\}
$$

- H: propagation hops (default: 2)
- γ: decay factor (default: 0.5)

### 3.4 Phase 3: Hierarchical Adaptive Weighting

#### 3.4.1 Level 1: Epoch-wise Adaptation

**Decay rate monitoring**:
$$
r_k(t) = \frac{\mathcal{L}_k(t) - \mathcal{L}_k(t-\Delta t)}{\Delta t}
$$

**Imbalance detection**:
$$
\text{imbalance} = \frac{\max_k |r_k| - \min_k |r_k|}{\text{mean}_k |r_k|}
$$

**Weight adjustment**:
$$
\lambda_k^{(t+1)} = \lambda_k^{(t)} \cdot \exp(\eta \cdot \text{sign}(r_k - \bar{r}))
$$

#### 3.4.2 Level 2: Batch-wise Gradient Harmonization

**Gradient conflict detection**:
$$
\text{conflict}_{ij} = \frac{g_i \cdot g_j}{||g_i|| \cdot ||g_j||}
$$

**Correction factor**:
$$
\rho_i = \begin{cases}
1 & \text{if no conflict} \\
\text{softplus}(1 - \min_j \text{conflict}_{ij}) & \text{otherwise}
\end{cases}
$$

### 3.5 Phase 4: Multi-physics Constraints

**Total loss** (fully unsupervised):

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}
$$

where:
- $\mathcal{L}_{\text{PDE}} = \frac{1}{N} \sum_{i=1}^N \tilde{w}_i (Au_i - b_i)^2$
- $\mathcal{L}_{\text{BC}}$: Dirichlet + Neumann BC
- $\mathcal{L}_{\text{IC}}$: Initial condition (time-dependent)
- $\mathcal{L}_{\text{cons}}$: Conservation laws (mass, momentum, energy)

All λ weights adapted by hierarchical weighting (Phase 3)

### 3.6 Phase 5: Curriculum Learning

**Hybrid loss** (supervised → unsupervised):

$$
\mathcal{L}_{\text{hybrid}} = \lambda_{\text{data}}(t) \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{physics}}
$$

**Schedule**:
$$
\lambda_{\text{data}}(t) = \lambda_{\text{init}} \cdot \exp(-\alpha \cdot t)
$$

---

## 4. Experimental Setup

### 4.1 Datasets

- **Cavity Flow**: Lid-driven cavity (Re = 100, 1000)
- **Backward-facing Step**: Separated flow
- **Cylinder Flow**: Vortex shedding

Mesh quality variations:
- High quality: skewness < 0.3
- Medium quality: 0.3 < skewness < 0.6
- Low quality: skewness > 0.6

### 4.2 Baselines

1. **Fixed-weight PINNs**: Uniform λ = 1.0
2. **Gradient normalization**: [Chen et al., 2018]
3. **Uncertainty weighting**: [Kendall et al., 2018]
4. **lbPINNs**: [Wang et al., 2021]
5. **Supervised GNN**: With ground truth

### 4.3 Evaluation Metrics

**Physics-based** (no ground truth required):
- PDE residual: ||Au - b||
- BC error: ||u_boundary - u_BC||
- Conservation: ||∇·u||

**Reference-based** (for validation):
- MSE, MAE, Relative error
- Max error

### 4.4 Implementation Details

- Framework: PyTorch + PyTorch Geometric
- Model: SimpleSAGE (4 layers, 64 hidden)
- Optimizer: Adam (lr=1e-3)
- Epochs: 200 (unsupervised), 100 (supervised)
- Hardware: NVIDIA GPU

---

## 5. Results and Analysis

### 5.1 Main Results

**Table 1: Comparison with Baselines**

| Method | High Quality Mesh | Medium Quality | Low Quality |
|--------|-------------------|----------------|-------------|
| | PDE Residual / BC Error / MSE | | |
| Fixed-weight | X / Y / Z | | |
| Grad norm | | | |
| Uncertainty | | | |
| lbPINNs | | | |
| **PUP-HAW-U** | **X / Y / Z** | | |

**Key findings**:
- PUP-HAW-U achieves X% improvement on low-quality meshes
- Competitive with supervised learning on high-quality meshes
- Faster convergence due to adaptive weighting

### 5.2 Mesh Quality Dependency

**Figure 1**: Performance vs. mesh quality
- X-axis: Average mesh skewness
- Y-axis: PDE residual
- PUP-HAW-U shows robust performance across quality ranges

### 5.3 Training Dynamics

**Figure 2**: Loss curves
- PUP-HAW-U converges faster (fewer epochs)
- More stable training (less oscillation)

### 5.4 Weight Distribution Analysis

**Figure 3**: Learned weight distribution
- Higher weights in high-skewness regions
- Correlation between weights and solution curvature

---

## 6. Ablation Study

**Table 2: Component Contributions**

| Configuration | PDE Residual | BC Error | Conservation | MSE |
|--------------|--------------|----------|--------------|-----|
| Full (PUP-HAW-U) | **X** | **Y** | **Z** | **W** |
| w/o Physics weights | | | | |
| w/o Hierarchical | | | | |
| w/o Topology prop | | | | |
| w/o Multi-physics | | | | |
| Baseline | | | | |

**Analysis**:
- Physics-based weights: XX% improvement
- Hierarchical adaptation: YY% improvement
- Topology propagation: ZZ% improvement
- All components contribute significantly

---

## 7. Conclusion

### 7.1 Summary

We proposed PUP-HAW-U, a novel framework for unsupervised PDE solving that:
1. Adapts to mesh quality variations via physics-informed weighting
2. Automatically balances multiple objectives via hierarchical adaptation
3. Achieves competitive accuracy without ground truth data

### 7.2 Limitations

- Assumes known PDE operators (A, b)
- Boundary types must be specified or detected
- Computational overhead from adaptive mechanisms

### 7.3 Future Work

- Extension to time-dependent PDEs
- Coupling with mesh refinement
- Application to multi-phase flows
- Uncertainty quantification

---

## Appendix

### A. Implementation Details

- Code: https://github.com/uzawa-ken/GNNtest2
- Reproducibility: All scripts and configs provided

### B. Additional Experiments

- Sensitivity to hyperparameters
- Generalization to unseen geometries
- Scalability analysis

### C. Mathematical Proofs

- Convergence guarantees (if applicable)
- Theoretical analysis of weighting schemes

---

## References

[To be filled with actual citations]

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks
2. Pfaff, T., et al. (2021). Learning mesh-based simulation with graph networks
3. Sanchez-Gonzalez, A., et al. (2020). Learning to simulate complex physics with graph networks
4. Chen, Z., et al. (2018). GradNorm: Gradient normalization for adaptive loss balancing
5. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty
6. Wang, S., et al. (2021). When and why PINNs fail to train: A neural tangent kernel perspective

---

**Target Journals/Conferences**:
- Journal of Computational Physics (JCP)
- Computer Methods in Applied Mechanics and Engineering (CMAME)
- NeurIPS (Machine Learning track)
- ICML (Physics ML workshop)
- ICLR

**Estimated Length**: 15-20 pages (double column)

**Timeline**:
- Experiments: 2-3 weeks
- Writing: 2-3 weeks
- Revision: 1-2 weeks
- **Total**: 5-8 weeks to submission
