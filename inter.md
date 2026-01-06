# 📄 Interviewer Handout

### Transformer Attention — Whiteboard Exercise (Single Head)

**Goal**
Assess the candidate’s ability to reason about attention from first principles using linear algebra, shapes, and clear explanation.

**What this is NOT**

* Not a PyTorch / TensorFlow API test
* Not about backprop or gradients
* Not about memorizing the paper

**What this IS**

* Shape reasoning
* Linear algebra fluency
* Ability to explain attention clearly
* Conceptual correctness

---

## Constraints (state explicitly)

Write this at the top of the whiteboard:

```
• Single-head self-attention
• Batch size: B
• Sequence length: L
• Embedding dimension: D
• No positional encodings
• Forward pass only
```

---

## Interview Structure (45 minutes total)

| Phase | Topic               | Time      |
| ----- | ------------------- | --------- |
| 1     | Setup & projections | 10–12 min |
| 2     | Attention scores    | 8–10 min  |
| 3     | Scaling & softmax   | 8–10 min  |
| 4     | Output computation  | 8–10 min  |
| 5     | Sanity / extension  | 5 min     |

---

## Phase 1 — Setup & Projections

**Prompt**

> “We start with an embedded input sequence
> ( X ∈ ℝ^{B × L × D} ).
> Show how to compute queries, keys, and values.”

**Expected derivation**

```
W_Q, W_K, W_V ∈ ℝ^{D × D}

Q = X W_Q
K = X W_K
V = X W_V
```

**Expected shapes**

```
Q, K, V ∈ ℝ^{B × L × D}
```

**What you’re looking for**

* Correct shape propagation
* Clear explanation of *why* Q, K, V are separate
* Comfort ignoring batch temporarily (good sign)

---

## Phase 2 — Attention Scores

**Prompt**

> “For a single batch element, compute the attention scores.
> What operation do we use and what shape do we get?”

**Expected**

```
Fix b:

Q ∈ ℝ^{L × D}
K ∈ ℝ^{L × D}

S = Q Kᵀ ∈ ℝ^{L × L}
```

**Interpretation**

```
S_ij = ⟨ q_i , k_j ⟩
```

Candidate should say something like:

> “This computes similarity between every query token and every key token.”

🚩 Red flag if they can’t explain why this is L×L.

---

## Phase 3 — Scaling & Softmax

**Prompt**

> “How do we convert scores into probabilities?
> Why do we scale?”

**Expected**

```
S_scaled = S / √D
A = softmax(S_scaled, axis=1)
```

**Properties**

```
A ∈ ℝ^{L × L}
∑_j A_ij = 1
```

**Acceptable intuition**

* Prevents large dot products
* Keeps softmax gradients stable

🚩 Red flag: “because the paper says so”

---

## Phase 4 — Output Computation

**Prompt**

> “How do we compute the output representations?”

**Expected**

```
Y = A V
```

**Shapes**

```
(L × L) @ (L × D) → (L × D)
```

**Expanded**

```
y_i = Σ_j A_ij · v_j
```

💡 This sentence is gold:

> “Each output token is a weighted sum of value vectors.”

---

## Phase 5 — Optional Extension (pick ONE)

Choose based on time:

* “Where would causal masking apply?”
* “What changes for cross-attention?”
* “How does multi-head attention differ conceptually?”
* “Why is attention permutation-equivariant?”

Do **not** ask multiple.

---

# 🧠 Expected Whiteboard Diagram (Canonical)

This is what a *strong* candidate’s board typically converges to.

```
X : (B, L, D)
│
├── W_Q ──▶ Q : (B, L, D)
├── W_K ──▶ K : (B, L, D)
└── W_V ──▶ V : (B, L, D)

(for one batch)

Q : (L, D)
K : (L, D)
V : (L, D)

S = Q Kᵀ
S : (L, L)

A = softmax(S / √D)
A : (L, L)

Y = A V
Y : (L, D)
```

If they draw arrows and label shapes clearly → **strong signal**.

---

# 📊 Grading Rubric (Simple & Effective)

| Category          | Strong                | Weak               |
| ----------------- | --------------------- | ------------------ |
| Shape reasoning   | Immediate, consistent | Frequent confusion |
| Linear algebra    | Clean, correct        | Hand-wavy          |
| Explanation       | Intuitive & precise   | Symbol dumping     |
| Scaling intuition | Correct               | Memorized          |
| Time management   | Finishes              | Gets stuck early   |

---

## Strong Candidate Signals (Green Flags)

* “Let’s ignore batch for clarity”
* Writes shapes unprompted
* Explains dot products intuitively
* States weighted-sum interpretation
* Notices quadratic complexity

## Weak Signals (Red Flags)

* Confuses L vs D repeatedly
* Cannot explain softmax axis
* Treats attention as magic
* Needs hints for every step

---

# 🧩 Interviewer Rescue Prompts (Use Sparingly)

If candidate stalls:

* “What shape do you want the output to be?”
* “How many tokens are interacting here?”
* “Think pairwise similarities.”

These preserve signal without giving answers.

---

## Final Interviewer Guidance

Say this once at the start:

> “This is collaborative — I’m evaluating your reasoning, not speed.”

That one sentence dramatically improves candidate performance **without reducing signal**.

--- 

Complete NUmeric Example

---

# 🔢 Numeric Attention Example (Single-Head, Tiny Sentence)

## Setup (you write this on the board)

> Sentence length **L = 2**
> Embedding dimension **D = 2**
> Single-head self-attention
> Ignore batch dimension

Sentence (purely symbolic):

```
["I", "code"]
```

---

## Step 1 — Input embeddings

Give them **explicit numbers** (critical for timing):

```
X =
[ 1  0 ]   ← token 1 ("I")
[ 0  1 ]   ← token 2 ("code")
```

So:

```
X ∈ ℝ^{2 × 2}
```

---

## Step 2 — Projection matrices (identity on purpose)

Tell the candidate:

> “Let’s make the projections trivial so we can focus on attention.”

```
W_Q = W_K = W_V = I₂
```

Therefore:

```
Q = X
K = X
V = X
```

Still:

```
Q, K, V ∈ ℝ^{2 × 2}
```

---

## Step 3 — Compute attention scores

Ask:

> “Compute the attention score matrix S = QKᵀ.”

They should compute:

```
Kᵀ =
[ 1  0 ]
[ 0  1 ]

S = QKᵀ =
[ 1·1 + 0·0   1·0 + 0·1 ] = [ 1  0 ]
[ 0·1 + 1·0   0·0 + 1·1 ]   [ 0  1 ]
```

So:

```
S =
[ 1  0 ]
[ 0  1 ]
```

Shape check:

```
S ∈ ℝ^{2 × 2}
```

Interpretation (they should say this):

* Token 1 attends most to itself
* Token 2 attends most to itself

---

## Step 4 — Scaling

Tell them explicitly:

```
√D = √2 ≈ 1.414
```

Scaled scores:

```
S_scaled =
[ 1/√2   0     ]
[ 0      1/√2  ]
≈
[ 0.707  0     ]
[ 0      0.707 ]
```

---

## Step 5 — Softmax (row-wise)

Ask:

> “Apply softmax row by row.”

Row 1:

```
softmax([0.707, 0]) =
[ e^0.707 / (e^0.707 + 1),
  1        / (e^0.707 + 1) ]
≈ [0.67, 0.33]
```

Row 2:

```
softmax([0, 0.707]) ≈ [0.33, 0.67]
```

So attention matrix:

```
A =
[ 0.67  0.33 ]
[ 0.33  0.67 ]
```

Key property to emphasize:

```
Each row sums to 1
```

---

## Step 6 — Output computation

Ask:

> “Now compute Y = A V.”

Recall:

```
V =
[ 1  0 ]
[ 0  1 ]
```

Row 1:

```
y₁ = 0.67·[1,0] + 0.33·[0,1]
   = [0.67, 0.33]
```

Row 2:

```
y₂ = 0.33·[1,0] + 0.67·[0,1]
   = [0.33, 0.67]
```

Final output:

```
Y =
[ 0.67  0.33 ]
[ 0.33  0.67 ]
```

Shape:

```
Y ∈ ℝ^{2 × 2}
```

---

## Step 7 — Interpretation (this is the signal)

A strong candidate will say:

> “Each output token is a weighted mixture of both value vectors, biased toward itself.”

If they say this unprompted — **excellent signal**.

---

## Optional follow-ups (pick ONE)

If time remains:

1. **Causal mask**

   * Mask upper-right element of `S`
   * Ask what changes in `A`

2. **Change embeddings**

   * Make both tokens identical
   * Ask what happens to attention

3. **Why this isn’t just averaging**

   * Let them explain adaptivity

---

## Why this example works

✅ Minimal arithmetic
✅ No matrix larger than 2×2
✅ Exercises **every step of attention**
✅ Easy to debug mistakes
✅ Whiteboard-friendly
✅ Scales naturally to multi-head discussion


---

# Transformer Self-Attention — Theory Worksheet (with Full Answers)

> **Single-head self-attention, forward pass only**
> This document contains the *fully worked theoretical derivation* and is intended for **interviewer reference**.

---

## Assumptions

* Single-head self-attention
* Batch size: ( B )
* Sequence length: ( L )
* Embedding dimension: ( D )
* No positional encodings
* No gradients or backpropagation

---

## 1. Input Representation

Let the embedded input sequence be:

$$
\mathbf{X} \in \mathbb{R}^{B \times L \times D}
$$

Each batch element contains a sequence of ( L ) tokens, each represented by a ( D )-dimensional embedding.

---

## 2. Linear Projections (Queries, Keys, Values)

We introduce three learned projection matrices:

$$
\mathbf{W}_Q,; \mathbf{W}_K,; \mathbf{W}_V \in \mathbb{R}^{D \times D}
$$

The query, key, and value tensors are computed as:

$$
\mathbf{Q} = \mathbf{X} \cdot \mathbf{W}_Q
$$

$$
\mathbf{K} = \mathbf{X} \cdot \mathbf{W}_K
$$

$$
\mathbf{V} = \mathbf{X} \cdot \mathbf{W}_V
$$

### Shapes

$$
\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times L \times D}
$$

Each token now has **three distinct representations**, enabling asymmetric similarity computation.

---

## 3. Attention Score Matrix

To simplify reasoning, fix a batch element and drop the batch dimension:

$$
\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times D}
$$

The attention score matrix is computed via matrix multiplication:

$$
\mathbf{S} = \mathbf{Q} \cdot \mathbf{K}^{\top}
$$

### Shape

$$
\mathbf{S} \in \mathbb{R}^{L \times L}
$$

### Element-wise Interpretation

$$
S_{ij} = \langle \mathbf{q}_i, \mathbf{k}_j \rangle
$$

Each entry represents the dot-product similarity between:

* the **query** vector of token ( i )
* the **key** vector of token ( j )

This computes **all pairwise token interactions**.

---

## 4. Scaling and Softmax

To stabilize the softmax operation, the score matrix is scaled:

$$
\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{D}}
$$

The attention matrix is then obtained by applying softmax **row-wise**:

$$
\mathbf{A} = \operatorname{softmax}!\left( \mathbf{S}_{\text{scaled}} \right)
$$

### Shape

$$
\mathbf{A} \in \mathbb{R}^{L \times L}
$$

### Key Property

For every query position ( i ):

$$
\sum_{j=1}^{L} A_{ij} = 1
$$

Each row of ( \mathbf{A} ) forms a **probability distribution over tokens**.

---

## 5. Output Computation

The output of attention is computed as:

$$
\mathbf{Y} = \mathbf{A} \cdot \mathbf{V}
$$

### Shape

$$
\mathbf{Y} \in \mathbb{R}^{L \times D}
$$

### Expanded Form

For each output token ( i ):

$$
\mathbf{y}*i = \sum*{j=1}^{L} A_{ij} , \mathbf{v}_j
$$

Each output vector is a **weighted sum of value vectors**.

---

## 6. Interpretation

* Each output token representation is **contextualized**
* Information from all tokens is incorporated
* Weights are determined dynamically via learned similarity
* The mechanism is **permutation-equivariant** (in the absence of positional encoding)

---

## 7. One-Sentence Summary (Canonical)

> *Single-head self-attention computes all pairwise token similarities, normalizes them into probability distributions, and uses them to form weighted sums of value vectors, producing context-aware token representations.*


