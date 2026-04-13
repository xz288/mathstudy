<!-- topic: probability_estimate_theory | saved: 2026-04-13 19:51 -->
# Week 1 Notes: MLE & MAP Estimation

> **Study plan context:** Week 1, Day 1–4 — Estimation Theory  
> **Background note:** Physics bridges are marked with ⚛️

---

## Day 1–2: Likelihood Functions & Maximum Likelihood Estimation

### What is a likelihood function?

Given a model parameterized by **θ** and observed data **X = {x₁, x₂, ..., xₙ}**, the **likelihood** is:

```
L(θ; X) = p(X | θ) = ∏ᵢ p(xᵢ | θ)        (assuming i.i.d. samples)
```

**Key distinction:**
- `p(X | θ)` — read as a function of **X** with θ fixed → it's a probability distribution
- `L(θ; X)` — read as a function of **θ** with X fixed → it's a likelihood (not a probability; doesn't integrate to 1 over θ)

---

### Log-likelihood

In practice, always work with the **log-likelihood**:

```
ℓ(θ) = log L(θ; X) = ∑ᵢ log p(xᵢ | θ)
```

Why logs?
1. Products become sums → numerically stable, no floating-point underflow
2. log is monotone → maximizing ℓ(θ) is equivalent to maximizing L(θ)
3. Connects naturally to entropy and cross-entropy (Week 2)

---

### ⚛️ Physics Bridge: Likelihood as a Partition Function

This is a deep analogy worth sitting with.

In statistical mechanics, the **free energy** is:

```
F = -kT log Z(β)     where Z(β) = ∑ exp(-βEᵢ)
```

The log-partition function `log Z(β)` plays the same structural role as the log-likelihood `ℓ(θ)`. Maximizing log-likelihood is analogous to **minimizing free energy** — you're finding the parameter configuration that best "explains" the data, just as the equilibrium state minimizes free energy.

More concretely: if you write `p(xᵢ | θ) = exp(θᵀφ(xᵢ) - A(θ))` (exponential family form), then `A(θ) = log Z(θ)` is exactly the log-partition function.

---

### MLE: The Recipe

**Maximum Likelihood Estimation** finds:

```
θ_MLE = argmax_θ ℓ(θ) = argmax_θ ∑ᵢ log p(xᵢ | θ)
```

Procedure:
1. Write down the log-likelihood for your model
2. Take the derivative with respect to θ and set to zero: `∂ℓ/∂θ = 0`
3. Solve for θ (closed form when possible, gradient ascent otherwise)

---

### MLE Derivations

#### Gaussian (known variance σ², estimate mean μ)

```
p(xᵢ | μ) = (1/√(2πσ²)) exp(-(xᵢ - μ)²/2σ²)

ℓ(μ) = -n/2 log(2πσ²)  -  1/(2σ²) ∑ᵢ (xᵢ - μ)²

∂ℓ/∂μ = 1/σ² ∑ᵢ (xᵢ - μ) = 0

→  μ_MLE = (1/n) ∑ᵢ xᵢ   (the sample mean)
```

For variance (with μ known):
```
σ²_MLE = (1/n) ∑ᵢ (xᵢ - μ)²
```
Note: this is **biased** (divides by n, not n-1) — relevant for Day 5–6.

---

#### Bernoulli (coin flips, estimate probability p)

```
p(xᵢ | p) = pˣⁱ (1-p)^(1-xᵢ),    xᵢ ∈ {0,1}

ℓ(p) = (∑ xᵢ) log p  +  (n - ∑ xᵢ) log(1-p)

∂ℓ/∂p = (∑ xᵢ)/p  -  (n - ∑ xᵢ)/(1-p) = 0

→  p_MLE = (1/n) ∑ᵢ xᵢ   (fraction of heads)
```

---

#### Poisson (count data, estimate rate λ)

```
p(xᵢ | λ) = e^(-λ) λˣⁱ / xᵢ!

ℓ(λ) = -nλ  +  (∑ xᵢ) log λ  -  ∑ log(xᵢ!)

∂ℓ/∂λ = -n + (∑ xᵢ)/λ = 0

→  λ_MLE = (1/n) ∑ᵢ xᵢ   (sample mean — makes sense for Poisson)
```

**Pattern:** For all three, MLE = the sample mean of the sufficient statistic. This is not a coincidence — it's a property of exponential families (Week 3).

---

### MLE as Minimizing Cross-Entropy

A useful reframe: MLE over n samples is equivalent to minimizing the **empirical cross-entropy** between the data distribution p̂ and the model p(· | θ):

```
θ_MLE = argmin_θ  -1/n ∑ᵢ log p(xᵢ | θ)  =  argmin_θ  H(p̂, p_θ)
```

This is exactly the **cross-entropy loss** used in classification. Training a classifier with cross-entropy *is* doing MLE. (Full development in Week 2.)

---

## Day 3–4: MAP Estimation & Priors

### From MLE to MAP

MLE ignores any prior knowledge about θ. **Maximum A Posteriori (MAP)** estimation incorporates a prior `p(θ)` via Bayes' rule:

```
p(θ | X) ∝ p(X | θ) · p(θ)
              ↑            ↑
          likelihood      prior
```

MAP finds the mode of the posterior:

```
θ_MAP = argmax_θ  log p(X | θ) + log p(θ)
                  ─────────────   ─────────
                  log-likelihood  log-prior
```

**MLE is a special case of MAP** with a flat (uniform) prior: `log p(θ) = const`.

---

### MAP = MLE + Regularizer

The log-prior acts as a **regularization term** on the log-likelihood. Two canonical cases:

#### Gaussian Prior → L2 Regularization (Ridge)

Place a zero-mean Gaussian prior on θ:

```
p(θ) = N(0, τ²I)

log p(θ) = -1/(2τ²) ‖θ‖²  +  const
```

MAP objective:

```
θ_MAP = argmax_θ  ∑ᵢ log p(xᵢ | θ)  -  λ‖θ‖²     where λ = 1/(2τ²)
```

This is exactly **ridge regression** (L2 regularization). A tighter prior (smaller τ²) → larger λ → stronger shrinkage toward zero.

---

#### Laplace Prior → L1 Regularization (Lasso)

Place a Laplace prior on θ:

```
p(θ) = ∏ⱼ (1/2b) exp(-|θⱼ|/b)

log p(θ) = -1/b ‖θ‖₁  +  const
```

MAP objective:

```
θ_MAP = argmax_θ  ∑ᵢ log p(xᵢ | θ)  -  λ‖θ‖₁     where λ = 1/b
```

This is **lasso regression** (L1 regularization). The Laplace prior has heavier tails than Gaussian but a sharper peak at zero, which is why L1 encourages **sparsity** — it's more willing to set parameters exactly to zero.

---

### Geometric Intuition: Why L1 is Sparse, L2 is Not

```
L2 ball (sphere): smooth, no corners
  → the MLE solution minus the L2 constraint rarely lands exactly on an axis

L1 ball (diamond): has corners at the axes
  → the constrained optimum often hits a corner, where one coordinate = 0
```

This is why lasso produces sparse solutions and ridge doesn't — it's a geometric consequence of the prior shape, not a magic property of L1.

---

### Choosing a Prior: Conjugate Priors

A **conjugate prior** is one where the posterior has the same functional form as the prior, making MAP (and full Bayesian inference) analytically tractable.

| Likelihood   | Conjugate Prior | Posterior       |
|--------------|-----------------|-----------------|
| Bernoulli    | Beta(α, β)      | Beta            |
| Poisson      | Gamma(α, β)     | Gamma           |
| Gaussian (μ) | Gaussian        | Gaussian        |
| Multinomial  | Dirichlet       | Dirichlet       |

**Example — Beta-Bernoulli:**

```
Prior:     p(p) = Beta(α, β)  ∝  p^(α-1) (1-p)^(β-1)
Likelihood: p(X|p) ∝  p^(∑xᵢ) (1-p)^(n-∑xᵢ)

Posterior: p(p|X) = Beta(α + ∑xᵢ,  β + n - ∑xᵢ)

MAP: p_MAP = (α + ∑xᵢ - 1) / (α + β + n - 2)
MLE: p_MLE = ∑xᵢ / n
```

The prior counts (α, β) act as **pseudo-observations** — a clean Bayesian interpretation of regularization. With α = β = 1 (uniform prior), MAP = MLE.

---

### ⚛️ Physics Bridge: Prior as a Constraint / Effective Hamiltonian

In a physics framing, the MAP objective:

```
θ_MAP = argmax_θ  ℓ(θ) + log p(θ)
```

is equivalent to finding the ground state of an effective Hamiltonian:

```
H_eff(θ) = -ℓ(θ) - log p(θ)
```

The prior `p(θ)` introduces an "energy penalty" for unlikely parameter values. A Gaussian prior is a harmonic potential; a Laplace prior is an L1 potential. This is the same structure as adding a regularization term to a Lagrangian.

In the limit of infinite data, the likelihood dominates and the prior becomes irrelevant — exactly as a coupling constant becomes negligible at high energies in an RG flow.

---

### Summary: MLE vs MAP vs Full Bayes

| Method      | What it returns       | Prior used | Computation |
|-------------|-----------------------|------------|-------------|
| MLE         | Point estimate (mode of likelihood) | None (flat) | Easy |
| MAP         | Point estimate (mode of posterior)  | Yes        | Easy |
| Full Bayes  | Full posterior distribution         | Yes        | Hard (usually) |

MAP is the practical middle ground: it incorporates prior knowledge with no more computational cost than MLE. Full Bayes is better in principle but requires integration over θ — this motivates variational inference (Week 2) and MCMC (Week 3).

---

## Key Takeaways

1. **MLE = argmax of log-likelihood** — for exponential family distributions, always yields the sample mean of the sufficient statistic.

2. **Log-likelihood connects to free energy** — maximizing ℓ(θ) is minimizing a free energy; the log-partition function is structurally identical.

3. **MAP = MLE + log-prior** — adding a prior is just adding a regularization term to the objective.

4. **Prior shape → regularizer shape:** Gaussian prior ↔ L2 (ridge), Laplace prior ↔ L1 (lasso).

5. **Regularization strength λ = 1/(prior variance)** — a more concentrated prior means stronger regularization.

6. **Cross-entropy loss in ML = negative log-likelihood** — training classifiers is MLE, always.

---

*Next up: Day 5–6 — Estimator properties (bias, variance, consistency, Cramér-Rao bound)*
