# How a Vision Network Sees Legs: Layer by Layer

## 1. The big picture

```
  Classical CV          CNN                    Vision Transformer        Foundation model
  (OpenCV)              (learned filters)      (adaptive attention)      (SAM, Depth Anything)
  hand-made kernel  ->  kernels learned   ->   weights computed     ->   giant ViT + huge data
  Sobel, Gaussian       by gradient descent    from the content          + promptable, zero-shot
  
  Underneath ALL of them: multiply-and-add = matrix multiplication
```

## 2. CNN: where the leg appears

```
 INPUT IMAGE (lion)
      |
      v
 +-----------------------------+
 | EARLY LAYERS (1-2)          |   edges at angles, colour blobs, gradients
 |  /  |  \  --   o            |   ~ learned Sobel / Gabor filters
 |  SAME for cat and lion      |
 +-----------------------------+
      |
      v
 +-----------------------------+
 | MIDDLE LAYERS               |   fur texture, parallel lines, joints, paw tips
 |   ||   +  fur  +  paw tip   |   --> a "LEG-LIKE" detector emerges here
 |  still SHARED cat / lion    |
 +-----------------------------+
      |
      v
 +-----------------------------+
 | DEEP LAYERS                 |   parts in context:
 |  leg + body + mane + grass  |   leg attached to body, face, mane, background
 |  paths start to SPLIT       |
 +-----------------------------+
      |
      v
 +-----------------------------+
 | FINAL LAYER (classifier)    |   combine all evidence
 |   lion 0.94   cat 0.03 ...  |
 +-----------------------------+
```

## 3. Cat leg vs lion leg

```
            leg detector fires for BOTH
                     |
        +------------+------------+
        |                         |
   CONTEXT CUES              CONTEXT CUES
   small vs sofa             large vs grass
   indoor scene              savanna scene
   no mane                   mane nearby
   varied fur                tawny short fur
        |                         |
        v                         v
       CAT                       LION

 The difference lives in DEEP layers + context, not in the leg itself.
```

## 4. Vision Transformer version

```
 Image -> cut into 16x16 patches -> each patch = a token

 EARLY layers:  patch attends mostly to NEIGHBOURS   (acts like local filters)
   [p1]<->[p2]   [p5]<->[p6]

 MIDDLE layers: patches GROUP into parts
   leg patches attend to each other  -> "this is a leg"

 LATE layers:   GLOBAL attention
   leg <-> head <-> mane <-> background  -> "lion"

 Each patch makes:
   Query  = what am I looking for?
   Key    = what do I contain?
   Value  = what I share if matched
   attention = softmax(Q K^T / sqrt(d)) V
```

## 5. Two caveats

- Usually no single neat "leg neuron": concepts are distributed across many units, and one unit can respond to several unrelated things.
- You can still look inside: feature visualisation, attention maps, linear probes.

## 6. Experiment to try

1. Load DINOv2 from the Hugging Face Hub.
2. Feed it a lion photo.
3. Plot the attention maps per layer and per head.
4. Watch early layers show edges, middle layers light up legs and head, late layers cover the whole animal, with no labels ever given.

## 7. CLIP: the image–text matching game

CLIP trains two encoders so matching images and captions land close together in one shared space.

```
  image  --> [ Image encoder (ViT/CNN) ] --> I  (vector)
  caption --> [ Text encoder (transformer) ] --> T  (vector)
                         both in the SAME embedding space
```

Similarity grid for a batch of N pairs (score = I · T):

```
                 T1:"a lion   T2:"a cat    T3:"a zebra   T4:"a red
                  in grass"   on a sofa"   at a lake"    car"
  I1 (lion)     [ HIGH ]       low          low           low
  I2 (cat)        low        [ HIGH ]       low           low
  I3 (zebra)      low          low        [ HIGH ]        low
  I4 (car)        low          low          low         [ HIGH ]

  Diagonal = correct pairs  -> push UP
  Off-diagonal = mismatches -> push DOWN
  Done both ways: each row picks its caption, each column picks its image
  Real training: N ~ 32,000 per batch, ~400 million pairs total
```

The loss, per row and per column (temperature τ):

```latex
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{I_i\cdot T_i/\tau}}{\sum_{j} e^{I_i\cdot T_j/\tau}} + \log\frac{e^{I_i\cdot T_i/\tau}}{\sum_{j} e^{I_j\cdot T_i/\tau}}\right]
```

Zero-shot classification afterwards:

```
  new photo --> Image encoder --> I

  "a photo of a lion"    --> T_lion     I · T_lion    = 0.31  <-- winner
  "a photo of a cat"     --> T_cat      I · T_cat     = 0.18
  "a photo of a cheetah" --> T_cheetah  I · T_cheetah = 0.22

  New class? Just type a new prompt. No retraining.
```

- Weak spots: counting, spatial relations (left of / right of), very fine-grained species, web biases.
- Where it lives on: text side of Stable Diffusion, OWL-ViT style detectors, vision encoders in multimodal LLMs.

## 8. From mathematics to silicon

Modern AI reduces to dense matrix multiplication, and every layer of the stack exists to do that faster.

| Layer | What it is | Key object |
| --- | --- | --- |
| Mathematics | Approximate an unknown function or distribution from samples | Loss, gradient, KL divergence |
| Model | Composition of affine maps + nonlinearities; attention = learned kernel smoothing | Matrices of weights |
| Algorithm | SGD + backprop (reverse-mode chain rule) | Matrix multiply + pointwise ops |
| Software | PyTorch / JAX compile the graph to kernels | Tensor kernels (CUDA, XLA) |
| Chip architecture | Thousands of multiply-accumulate units fed by fast memory | Systolic array / tensor core |
| Digital logic | Adders, multipliers, registers, clock | MAC unit |
| Physics | CMOS transistors switching at \~1–2 GHz | Transistor (classical, not quantum) |

## 9. Inside an AI chip: transistor to rack

An AI chip is a classical CMOS processor (an ASIC like Google's TPU, or a GPU), not storage, not an FPGA, not quantum. It has tens to hundreds of billions of transistors.

```
 LEVEL 0  TRANSISTOR        CMOS switch, ~3-5 nm process, on/off = 1/0
             |
 LEVEL 1  LOGIC GATES       NAND, NOR, XOR  (a few transistors each)
             |
 LEVEL 2  ARITHMETIC        half adder = XOR + AND
                            full adder = 2 half adders + OR
                            multiplier = grid/tree of full adders
             |
 LEVEL 3  MEMORY + TIME     flip-flops / registers hold values
                            clock (PLL, ~1-2 GHz) ticks every stage
             |
 LEVEL 4  MAC UNIT          a*b + acc -> acc
          (multiplier + adder + accumulator register)  = the atom of AI
             |
 LEVEL 5  SYSTOLIC ARRAY    grid of MACs, e.g. 128x128 or 256x256
          weights sit still, activations flow right, partial sums flow down
             |
 LEVEL 6  CORE              systolic array/tensor cores + vector unit
                            + scalar control + on-chip SRAM
             |
 LEVEL 7  DIE               many cores + network-on-chip + HBM controllers
                            + PCIe + chip-to-chip links (NVLink / ICI)
             |
 LEVEL 8  PACKAGE           compute die + HBM memory stacks
                            side by side on a silicon interposer
             |
 LEVEL 9  BOARD / MODULE    package + voltage regulators (~700-1000 W)
                            + heat sink or liquid cold plate
             |
 LEVEL 10 SERVER            ~8 accelerators + switch chips + host CPUs
                            + network cards + cooling
             |
 LEVEL 11 RACK              many servers, e.g. 72 GPUs in one rack,
                            ~100+ kW, liquid cooled
             |
 LEVEL 12 CLUSTER           thousands of racks linked by optical
                            InfiniBand / Ethernet = one giant computer
```

Systolic array, the heart of it:

```
  activations -->  [MAC]-->[MAC]-->[MAC]-->[MAC]
                     |       |       |       |
  activations -->  [MAC]-->[MAC]-->[MAC]-->[MAC]
                     |       |       |       |
  activations -->  [MAC]-->[MAC]-->[MAC]-->[MAC]
                     |       |       |       |
                     v       v       v       v
                   output partial sums (one matrix product)

  each tick: every MAC multiplies, adds, passes data to its neighbour
```

What it is and is not:

| Thing | Is an AI chip one? | Why |
| --- | --- | --- |
| Processor | Yes | It computes; specialised for matrix maths, weak at branching code |
| Storage | No, but it carries memory | On-chip SRAM + HBM beside it; memory bandwidth is the bottleneck |
| FPGA | Usually no | FPGAs can run AI but are reprogrammable and less efficient; AI chips are fixed silicon |
| Quantum chip | No | Classical transistors at room temperature; quantum chips are a separate field |

## 10. Architecture

Sections 10 to 13 compress this whole conversation into diagrams: the system, its parts, its vocabulary, then the maths.

### The whole system

[embedded content: the whole conversation as one system · 5 layers, 1 feedback loop]

Data lands on the Hub, models learn from it, applications use them, and silicon runs everything; pseudo-labels close the loop back to data.

### The lineage

[embedded content: lineage of vision models · 5 eras]

Every era still multiplies and adds; what changed is who sets the weights, from an engineer to data to the input itself.

## 11. Anatomy

### Transformer block

[embedded content: one transformer block · attention, MLP, two residual paths]

Each sub-layer only adds an update to its input, x + f(x), which is why stacking L blocks behaves like integrating a flow (section 13).

### Perception pipeline

[embedded content: LiDAR–camera fusion with foundation models · 2 sensors, 3 models, 1 fusion step]

Projection is the hinge: once each LiDAR point has a pixel, it can borrow a SAM label and lend its true distance to Depth Anything.

### AI chip package

[embedded content: AI chip package · HBM, die, core, and the path to a cluster]

Inside the core, data loops SRAM → systolic array → vector unit → SRAM; HBM sits beside the die because feeding the array is the bottleneck, not computing.

## 12. Terminology

[embedded content: terminology map · 8 concepts to 8 silicon units]

Only the matrix multiply row needs the giant MAC grid; every other row exists to keep that grid fed, controlled and connected.

## 13. Field theory

[embedded content: field-theory view · state, adjoint and parameter fields, then lattice and transistor]

Read the loop clockwise: φ runs forward through depth, the loss turns φ(T) into one number, ψ carries its sensitivity back, and θ moves a little. The chip then executes all of it as fields on a lattice.

**0. Data are fields.** An image is a field I: Ω ⊂ ℝ² → ℝ³; a LiDAR scan is an empirical measure μ = (1/N) Σ δ(x − xᵢ) on ℝ³. Calibration maps one onto the other:

```latex
u = \pi\big(K(Rx + t)\big), \qquad \pi(X, Y, Z) = \Big(\frac{X}{Z}, \frac{Y}{Z}\Big)
```

**1. Classical filtering is a fixed linear operator.** Convolution with a hand-chosen kernel k is diagonal in Fourier space:

```latex
\varphi = k * I \quad\Longleftrightarrow\quad \hat\varphi(\omega) = \hat k(\omega)\, \hat I(\omega)
```

**2. A CNN learns the kernels and composes them through a nonlinearity σ.**

```latex
\varphi_{\ell+1} = \sigma\big(k_\ell * \varphi_\ell + b_\ell\big), \qquad k_\ell \text{ learned from data}
```

**3. φ̇: a residual network is a flow in depth.** The update φₗ₊₁ = φₗ + h Fθ(φₗ) is forward Euler for an ODE in depth time t:

```latex
\dot\varphi = F_\theta(\varphi, t), \qquad t \in [0, T]
```

**4. Attention makes F depend on the whole field.** Tokens xᵢ act as particles, each moving with a softmax-weighted average of the others' values. With many tokens, their density μₜ is advected by a velocity it generates itself (idealised: no MLP, no normalisation):

```latex
\dot x_i = \sum_j \frac{e^{\langle Qx_i,\, Kx_j\rangle/\sqrt d}}{\sum_k e^{\langle Qx_i,\, Kx_k\rangle/\sqrt d}}\, V x_j
```

```latex
\partial_t \mu_t + \nabla\cdot\big(\mu_t\, v[\mu_t]\big) = 0, \qquad v[\mu](x) = \frac{\int e^{\langle Qx,\, Ky\rangle/\sqrt d}\, Vy\, d\mu(y)}{\int e^{\langle Qx,\, Ky\rangle/\sqrt d}\, d\mu(y)}
```

This velocity is not divergence-free, since tokens tend to cluster, so no literal stream function exists. The role of ψ passes to the adjoint.

**5. ψ̇: backprop is the adjoint equation.** Add the dynamics as a constraint with a multiplier field ψ. Making the Lagrangian stationary in φ gives a backward equation, exactly as in 4D-Var:

```latex
\mathcal L = \ell\big(\varphi(T)\big) + \int_0^T \psi^\top\big(F_\theta(\varphi, t) - \dot\varphi\big)\, dt
```

```latex
\dot\psi = -\Big(\frac{\partial F_\theta}{\partial \varphi}\Big)^{\!\top} \psi, \qquad \psi(T) = \nabla_\varphi\, \ell\big(\varphi(T)\big)
```

So ψ(t) = ∂ℓ/∂φ(t): the sensitivity of the loss to the state at depth t, integrated from T back to 0.

**6. θ̇ and δ: training is a gradient flow in parameter space.** In slow training time τ:

```latex
\frac{d\theta}{d\tau} = -\nabla_\theta \mathcal L = -\int_0^T \Big(\frac{\partial F_\theta}{\partial \theta}\Big)^{\!\top} \psi\, dt
```

SGD with step η is a noisy discretisation of this, approximately the SDE below. For a wide network, treat neurons as particles with density ρ over parameter space; training becomes a Wasserstein gradient flow driven by the first variation δ𝐻/δρ, which is where δ enters (Mei, Montanari and Nguyen 2018; Chizat and Bach 2018):

```latex
d\theta = -\nabla_\theta \mathcal L\, d\tau + \sqrt{\eta\, \Sigma(\theta)}\, dW_\tau \qquad \partial_\tau \rho = \nabla_\theta \cdot \Big(\rho\, \nabla_\theta \frac{\delta \mathcal L}{\delta \rho}\Big)
```

**7. What the loss measures.** An LLM factorises text into next-token conditionals, and cross-entropy equals the KL divergence to the data up to a constant. Loss then falls as a power law in parameters N and data D (Hoffmann et al. 2022):

```latex
p_\theta(x_{1:n}) = \prod_{k=1}^{n} p_\theta(x_k \mid x_{<k}), \qquad \mathbb E_{p_{\text{data}}}\big[-\log p_\theta\big] = \mathrm{KL}\big(p_{\text{data}} \,\|\, p_\theta\big) + H(p_{\text{data}})
```

```latex
L(N, D) \approx E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
```

**8. The chip discretises everything onto a lattice.** In a systolic array, activations a and partial sums s are two fields on a grid with clock k. Their continuum limit is a pair of transport equations, with the weights w as a source:

```latex
\partial_k a + \partial_x a = 0, \qquad \partial_k s + \partial_y s = w(x, y)\, a
```

**9. Underneath sits the electrostatic field.** Poisson's equation sets the potential V in each transistor channel. Power is spent charging capacitance C on every toggle, and long wires have large C, so moving data costs more than computing on it:

```latex
\nabla\cdot\big(\varepsilon \nabla V\big) = -\rho, \qquad P_{\text{dyn}} = \alpha\, C\, V^2 f
```

**Dictionary: data assimilation and deep learning**

| Data assimilation | Deep learning |
| --- | --- |
| Forward model, integrated in time | Network as a flow in depth, φ̇ = Fθ(φ) |
| Adjoint model, integrated backward | Backpropagation, ψ̇ |
| 4D-Var cost: background + observation misfit | Loss + weight decay, a Gaussian prior on θ |
| Control variable: initial state x₀ | Parameters θ |
| Analysis: conditional mean of the state | Attention output: kernel estimate of a conditional mean |
