<!-- topic: probability_estimate_theory | saved: 2026-04-16 01:04 -->
# Matrix Calculus: Two Essential Identities

> **Context:** Used in the MAP/ridge regression closed-form derivation.  
> These two identities are the matrix analogues of the scalar rules d/dx(bx) = b and d/dx(ax²) = 2ax.

---

## Setup: What does ∂/∂θ mean for a vector?

When θ is a p-dimensional vector, the gradient ∂f/∂θ is also a p-dimensional vector — each entry is the partial derivative with respect to one component:

```
∂f/∂θ  =  [ ∂f/∂θ₁,  ∂f/∂θ₂,  ...,  ∂f/∂θₚ ]ᵀ
```

The strategy for both proofs: write the scalar expression for a single component k, differentiate, then stack the results back into a vector.

---

## Identity 1: ∂/∂θ (bᵀθ) = b

### Scalar analogy

In 1D: d/dx(bx) = b. We're just generalizing this.

### Proof

bᵀθ is a dot product — a scalar:

```
bᵀθ  =  b₁θ₁ + b₂θ₂ + ... + bₚθₚ  =  Σᵢ bᵢθᵢ
```

Take the partial derivative with respect to the k-th component θₖ.  
Every term where i ≠ k has no θₖ in it, so it vanishes:

```
∂/∂θₖ (Σᵢ bᵢθᵢ)  =  bₖ
```

This holds for every k = 1, 2, ..., p. Stacking all p partials into a vector:

```
∂/∂θ (bᵀθ)  =  [ b₁, b₂, ..., bₚ ]ᵀ  =  b   ✓
```

**Note:** bᵀθ = θᵀb (dot product is commutative), so ∂/∂θ (θᵀb) = b as well.

---

## Identity 2: ∂/∂θ (θᵀAθ) = 2Aθ   (when A is symmetric)

### Scalar analogy

In 1D: d/dx(ax²) = 2ax. Here θᵀAθ is the matrix generalization of ax².

### Step 1 — Write out the double sum

θᵀAθ expands as:

```
θᵀAθ  =  Σᵢ Σⱼ θᵢ Aᵢⱼ θⱼ
```

where Aᵢⱼ is the (i,j) entry of A.

### Step 2 — Differentiate with respect to θₖ

Apply ∂/∂θₖ to the double sum. A term θᵢ Aᵢⱼ θⱼ depends on θₖ only when i = k or j = k (or both). Split into three cases:

**Case i = k, j ≠ k:**
```
∂/∂θₖ (θₖ Aₖⱼ θⱼ)  =  Aₖⱼ θⱼ
```

**Case i ≠ k, j = k:**
```
∂/∂θₖ (θᵢ Aᵢₖ θₖ)  =  Aᵢₖ θᵢ
```

**Case i = k, j = k:**
```
∂/∂θₖ (θₖ Aₖₖ θₖ)  =  ∂/∂θₖ (Aₖₖ θₖ²)  =  2Aₖₖ θₖ
```
(This is just the i=k, j≠k and i≠k, j=k cases both contributing, which we'll see combines naturally below.)

### Step 3 — Sum up all contributions

Collecting everything:

```
∂/∂θₖ (θᵀAθ)  =  Σⱼ Aₖⱼ θⱼ   +   Σᵢ Aᵢₖ θᵢ
                  ────────────       ────────────
                  k-th row of A      k-th col of A
                  dotted with θ      dotted with θ
```

In matrix notation, these two terms are:

```
Σⱼ Aₖⱼ θⱼ  =  (Aθ)ₖ          ← k-th entry of Aθ

Σᵢ Aᵢₖ θᵢ  =  (Aᵀθ)ₖ         ← k-th entry of Aᵀθ
```

So:

```
∂/∂θₖ (θᵀAθ)  =  (Aθ)ₖ  +  (Aᵀθ)ₖ  =  ((A + Aᵀ)θ)ₖ
```

### Step 4 — Apply symmetry of A

Stack all p components into a vector:

```
∂/∂θ (θᵀAθ)  =  (A + Aᵀ)θ
```

When A is **symmetric** (A = Aᵀ):

```
∂/∂θ (θᵀAθ)  =  2Aθ   ✓
```

### What if A is not symmetric?

You still get a valid formula — just not the clean 2Aθ form:

```
∂/∂θ (θᵀAθ)  =  (A + Aᵀ)θ       (general case)
```

In the ridge regression derivation, A = XᵀX. Is XᵀX symmetric?

```
(XᵀX)ᵀ  =  XᵀXᵀᵀ  =  XᵀX   ✓
```

Yes — XᵀX is always symmetric, so the 2Aθ identity applies cleanly.

---

## Putting it together: the ridge regression gradient

The objective was:

```
J(θ) = yᵀy  -  2θᵀXᵀy  +  θᵀXᵀXθ  +  λθᵀθ
```

Applying our two identities term by term:

```
∂/∂θ (yᵀy)        =  0                    (no θ)
∂/∂θ (-2θᵀXᵀy)    =  -2Xᵀy               (Identity 1, b = Xᵀy)
∂/∂θ (θᵀXᵀXθ)     =  2XᵀXθ              (Identity 2, A = XᵀX, symmetric)
∂/∂θ (λθᵀθ)       =  2λθ  =  2λIθ        (Identity 2, A = λI, symmetric)
```

Sum:

```
∂J/∂θ  =  -2Xᵀy  +  2XᵀXθ  +  2λIθ
         =  2(XᵀX + λI)θ  -  2Xᵀy
```

Set to zero → solve:

```
(XᵀX + λI)θ  =  Xᵀy

θ_MAP  =  (XᵀX + λI)⁻¹ Xᵀy   ✓
```

---

## Quick Reference

| Expression   | Gradient ∂/∂θ  | Condition        |
|--------------|----------------|------------------|
| bᵀθ          | b              | always           |
| θᵀb          | b              | always           |
| θᵀAθ         | (A + Aᵀ)θ     | always           |
| θᵀAθ         | 2Aθ            | A symmetric      |
| ‖θ‖²= θᵀθ   | 2θ             | (A = I, special case) |
