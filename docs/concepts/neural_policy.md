# Neural Policies and Policy Distillation

M2A trains small neural networks to reproduce the CFR strategy verified in M1.
This note explains the mathematics involved, and what the experiment can and
cannot show. The code is in `src/pokermind/neural/` and
`src/pokermind/experiments/neural_experiment.py`.

The research question is narrow: **can a tiny network reproduce a verified
CFR policy accurately enough to keep its exploitability low?** On Kuhn Poker
this tests whether the pieces fit together: encoding, masking, training,
serialization, evaluation and the agent interface. It is *not* a test of
generalization (§10).

## 1. Tensors and parameters

A **tensor** is an n-dimensional array of numbers: a scalar (0-d), vector
(1-d), matrix (2-d) and so on. PyTorch tensors also record how they were
computed, so gradients can be calculated automatically (§5).

A model's **parameters** `θ` are the tensors that training changes: the weight
matrices `W` and bias vectors `b` of each layer. Everything else (the inputs,
the targets, the mask) is fixed data. Training is the search for parameters
that make the model's outputs match the targets.

## 2. Encoding an information state

A network needs a numeric input vector `x`. PokerMind encodes an
`InfoState` explicitly, as 7 numbers:

| Index | Feature                     | Encoding            |
|-------|-----------------------------|---------------------|
| 0–2   | own card J, Q, K            | one-hot             |
| 3–6   | public history `""`, `x`, `b`, `xb` | one-hot     |

For example, `Q|xb` → `[0,1,0, 0,0,0,1]`.

The encoder accepts only an `InfoState`, which has no field for the
opponent's card, so the opponent's card can't reach the network. The seat is
not encoded separately because the history determines it.

The output covers all 4 actions in M0's enum order `[CHECK, BET, CALL, FOLD]`.
A **legal-action mask** `m ∈ {0,1}^4` marks the legal ones: `[1,1,0,0]` when
not facing a bet, `[0,0,1,1]` when facing one.

## 3. Layers and activations

A **linear (fully connected) layer** maps a vector `h ∈ ℝ^n` to `ℝ^k`:

```
z = W h + b,      W ∈ ℝ^{k×n}, b ∈ ℝ^k
```

Stacking linear layers alone is still linear, since `W₂(W₁h) = (W₂W₁)h`. A
nonlinear **activation** between them lets the network represent interactions
between features. PokerMind uses `tanh`, which is smooth, bounded and
zero-centred:

```
tanh(z) = (e^z − e^{−z}) / (e^z + e^{−z})
```

The three architectures tested:

| Name       | Shape          | Parameters |
|------------|----------------|-----------:|
| `linear`   | 7 → 4          | 32         |
| `mlp16`    | 7 → 16 → 4     | 196        |
| `mlp16x16` | 7 → 16 → 16 → 4| 468        |

## 4. Forward propagation, softmax and masking

**Forward propagation** computes the output from the input, one layer at a
time. For one hidden layer:

```
h = tanh(W₁ x + b₁)          hidden features
z = W₂ h + b₂                logits, one per action
```

**Softmax** turns the logits into a probability distribution. The legal-action
mask is applied *before* normalizing:

```
                  m_a · exp(z_a)
q_θ(a | x) = ─────────────────────────
              Σ_b m_b · exp(z_b)
```

In the code, illegal logits are set to −∞ (in practice the most negative
finite float). Their exponential is then exactly 0, so an illegal action gets
probability 0 regardless of the parameters, and also receives no gradient. The
result is non-negative and sums to 1 over the legal actions.

Softmax is unchanged when a constant is added to every logit, so only
*differences* between logits matter. With two legal actions,
`q(bet) = σ(z_bet − z_check)`, the logistic function of the logit gap.

## 5. Loss: cross-entropy and KL divergence

The teacher gives a target distribution `p(· | I)` for each of the `N = 12`
information sets. We train the student `q_θ` to match it by minimizing the
mean **Kullback–Leibler divergence**:

```
L(θ) = (1/N) Σ_I KL( p(·|I) ‖ q_θ(·|I) )
     = (1/N) Σ_I Σ_{a ∈ A(I)} p(a|I) · [ log p(a|I) − log q_θ(a|I) ]
```

`KL(p‖q) ≥ 0`, and it is 0 exactly when `q = p`. Expanding the bracket gives

```
KL(p ‖ q) = H(p, q) − H(p),
H(p, q)   = −Σ_a p(a) log q(a)     (cross-entropy)
H(p)      = −Σ_a p(a) log p(a)     (teacher entropy; does not depend on θ)
```

So minimizing KL and minimizing **soft-target cross-entropy** give identical
gradients. The code minimizes cross-entropy and reports KL, which is easier to
read because it reaches 0 at a perfect fit. Terms with `p(a) = 0` contribute
nothing (`0 · log 0 = 0`).

**Why train on full distributions rather than the most likely action?**
Kuhn's equilibrium is *mixed* (`docs/concepts/cfr.md` §4). Player 1 should
call a bet with Q with probability exactly 1/3. If we trained on the teacher's
argmax, the target would be "always fold". The resulting pure strategy is
exploitable: player 0 would bluff with J every time. The probabilities *are*
the strategy, and the indifference they create is what makes it
unexploitable. Distribution targets preserve them; argmax targets destroy
them.

### The gradient with respect to the logits

For softmax with cross-entropy there is a simple closed form:

```
∂H(p, q) / ∂z_a = q(a) − p(a)        for legal a
```

Each logit is pushed down in proportion to how much the student
*over*-predicts that action, and up when it under-predicts. The gradient is 0
when `q = p`.

## 6. Backpropagation

**Backpropagation** is the chain rule applied from the loss back through each
layer. For the one-hidden-layer network, with `δ = q − p` the logit gradient
above:

```
∂L/∂W₂ = δ hᵀ                    ∂L/∂b₂ = δ
∂L/∂h  = W₂ᵀ δ
∂L/∂u  = (∂L/∂h) ⊙ (1 − h²)      (u = W₁x + b₁, tanh'(u) = 1 − tanh²(u))
∂L/∂W₁ = (∂L/∂u) xᵀ              ∂L/∂b₁ = ∂L/∂u
```

`⊙` is element-wise multiplication. Each parameter's gradient says how a
small change to it would change the loss. PyTorch's autograd records the
forward computation and applies these rules automatically when
`loss.backward()` is called. The tests check autograd's gradient against a
finite-difference estimate.

## 7. Gradient descent and Adam

**Gradient descent** moves the parameters a small step against the gradient:

```
θ ← θ − η ∇_θ L(θ)          η = learning rate
```

Because all 12 examples fit in one batch, every step uses the exact gradient
of the full loss. That makes this full-batch gradient descent, with no
stochastic sampling.

**Adam** (Kingma & Ba, 2015) adapts the step size of each parameter. With
gradient `g_t` at step `t`:

```
m_t = β₁ m_{t−1} + (1 − β₁) g_t          running mean of gradients (momentum)
v_t = β₂ v_{t−1} + (1 − β₂) g_t²         running mean of squared gradients
m̂_t = m_t / (1 − β₁^t),  v̂_t = v_t / (1 − β₂^t)     bias correction
θ_t = θ_{t−1} − η · m̂_t / (√v̂_t + ε)
```

The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e−8`. Dividing by `√v̂`
makes the step size roughly `η` whatever the gradient's scale. That helps
here, because near-deterministic targets (for example "K always calls") ask
for large logits, which plain gradient descent approaches very slowly.

**One training step** in `neural/training.py`:

1. *Forward:* compute `q_θ` for all 12 states (§4).
2. *Loss:* cross-entropy against the teacher (§5).
3. *Zero gradients*, then `loss.backward()`: backpropagation fills `θ.grad`
   (§6).
4. `optimizer.step()`: the Adam update changes the parameters (§7).

Training is seeded and deterministic on the CPU. The parameters are
initialized from an explicit `torch.Generator`. The data has a fixed order
and there is no shuffling, so the same seed gives bit-identical weights.

## 8. Overfitting and generalization

**Overfitting** means fitting the training examples very well while doing
badly on new inputs from the same distribution. **Generalization** is
performance on inputs *not seen in training*, measured on a held-out set. A
large model can memorize a small dataset, so a low training loss on its own
proves nothing about generalization.

## 9. Supervised policy distillation

**Policy distillation** trains a student policy to imitate a teacher's
action distributions. Here the teacher is M1's CFR average strategy, which is
exactly evaluable and has a measured exploitability. The student never plays
during training. It is pure supervised learning with fixed targets. This is
*not* reinforcement learning: no rewards, no exploration, no self-play.

Imitation error doesn't map one-to-one onto exploitability. A small error at
an information set where the equilibrium depends on an indifference can cost
more than a larger error where one action clearly dominates. The experiment
therefore reports both, and flags any student whose exploitability is much
worse than its imitation error would suggest.

### Why the linear model can't fit the teacher

With the encoding of §2, a linear model's logit gap between BET and CHECK is
`α_card + β_history`: an additive card effect plus an additive history effect.
Player 0's root and player 1's "after a check" share the same two actions, so
they share `α_card`. The teacher's log-odds of betting are approximately:

| Card | P0 root (`""`) | P1 after check (`"x"`) | Difference |
|------|---------------:|-----------------------:|-----------:|
| J    | logit(0.21) ≈ −1.3 | logit(1/3) ≈ −0.7  | ≈ 0.6      |
| K    | logit(0.62) ≈ +0.5 | logit(≈1) ≫ +5     | ≫ 4        |

An additive model forces the "Difference" column to be the same for every
card, and the teacher needs very different values. So the linear student has
an **irreducible approximation error**: no amount of training removes it. A
hidden layer can represent the card × history interaction.

## 10. Why Kuhn Poker can't show generalization

Kuhn Poker has exactly **12 information sets**. The teacher dataset contains
each one exactly once, and nothing else exists. So:

* the training set *is* the entire input space, and no held-out information
  state is left to test on;
* repeating examples, or sampling the same states from simulated hands, adds
  no new evidence. It only re-weights the same 12 points;
* a 12-row lookup table already reproduces the teacher *exactly*, with no
  approximation error and no training.

The honest conclusion from M2A is therefore at most: "a network of size X can
**represent** this 12-state policy to within error Y, and the exact
exploitability of the result is Z". Whether neural policies help in poker only
becomes testable in games with far more information sets than parameters:
Leduc Hold'em, then Hold'em with abstraction. There, a held-out set of
information states can be compared against a table that can't be stored.
