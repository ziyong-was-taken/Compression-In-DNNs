# TOC

- [TOC](#toc)

## Networks

- supported models
  - MLP with custom widths and non-linearity
  - ConvNeXt-T
  - ResNet-18
- access activations $\{𝐡ˡ_{c,i}\}_{l∈ℒ}$ using `model.feature_extractor(`$𝐱_{c,i}$`)`
  - MLP: end of each non-linearity
  - ConvNeXt-T, ResNet-18: end of each residual block
- `model.get_encoder_decoder(`$l$`)` returns encoder up to layer $l$ and decoder starting from layer $l$
- compute metrics after each epoch using `on_train_epoch_end` callback of trainer (to access data)

## Metrics

### NC1

- goal: compute $\operatorname{tr}(Σ_W^l Σ_B^{l+})$ for all layers $l ∈ ℒ$
- issue: only batch activations $\{𝐡ˡ_{c,i}\}_{l ∈ ℒ,\ c ∈ \{1,…,C\},\ i ∈ \{1,…,S\}}$ fit in memory (batch size $S$)
- when training a batch, store running
  - class counts $\{n_c\}_{c=1}^C$
  - layer class sums $\{\{∑_{i=1}^{n_c} 𝐡_{c,i}^l\}_{c=1}^C\}_{l ∈ ℒ}$
  - gram matrix $G^l = ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l 𝐡_{c,i}^{l⊤}$
- $Σ_B^l$
  - use class counts and layer sums to compute $\{\boldsymbol μ_c^l\}_{l ∈ ℒ}$
  - sum class counts and sums to obtain $\{\bar{\boldsymbol μ}^l\}_{l ∈ ℒ}$
  - $Σ_B^l = 1/C ∑_{c=1}^C ({\boldsymbol μ}_c^l - \bar{\boldsymbol μ}^l)({\boldsymbol μ}_c^l - \bar{\boldsymbol μ}^l)^⊤$
- $Σ_W^l$
  - recall: $Σ_W^l + Σ_B^l = Σ_T^l = 1/N ∑_{c=1}^C ∑_{i=1}^{n_c}(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)^⊤ = Σ_W^l + Σ_B^l$
  - lemma: $Σ_T^l = G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤}$
    $$
    \begin{align*}
    Σ_T^l
      &= 1/N ∑_{c=1}^C ∑_{i=1}^{n_c}(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)^⊤ \\
      &= 1/N ∑_{c=1}^C ∑_{i=1}^{n_c} (𝐡_{c,i}^l 𝐡_{c,i}^{l⊤} - \bar{\boldsymbol μ}^l 𝐡_{c,i}^{l⊤} - 𝐡_{c,i}^l \bar{\boldsymbol μ}^{l⊤} + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤}) \\
      &= 1/N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l 𝐡_{c,i}^{l⊤} - 1/N ∑_{c=1}^C ∑_{i=1}^{n_c} (\bar{\boldsymbol μ}^l 𝐡_{c,i}^{l⊤} + 𝐡_{c,i}^l \bar{\boldsymbol μ}^{l⊤}) + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} \\
      &= G^l/N - \bar{\boldsymbol μ}^l\left(1/N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^{l⊤}\right) - \left(1/N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l\right) \bar{\boldsymbol μ}^{l⊤} + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} \\
      &= G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} \\
      &= G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤}
    \end{align*}
    $$
  - thus, $Σ_W^l = G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} - Σ_B^l$

### DIB

- compute new labels for all samples using Algorithm 1
- for each new label, create a copy of the decoder $D$ returned by `model.get_encoder_decoder()`
- combine encoder $E$ and decoders $D_1, D_2, \dots$ into single model $M$
  <!---->
  ```markdown
         E
  M =  / | \
      D₁ D₂ …
  ```
  <!---->
- after each training epoch, train $M$ using average cross-entropy loss over the decoder heads

### Rank
