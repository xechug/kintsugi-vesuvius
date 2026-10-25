# Kintsugi Physical XPCT: Multi-Scale Differential Phase-Contrast Operators, Differentiable Sheet Unwarping, and Structure-Tensor Fiber Suppression for Cross-Scroll Ink Detection in Carbonized Herculaneum Papyri

**Authors:** Jose M. Martin (`@xechug`) et al.  
**Repository:** [https://github.com/xechug/kintsugi-vesuvius](https://github.com/xechug/kintsugi-vesuvius) (MIT License)  
**Humanitarian Pledge:** 50% of any Vesuvius Challenge prize proceeds awarded to this research are pledged to cancer research.

---

## Abstract

Virtual unwrapping of carbonized Herculaneum papyrus scrolls via X-ray Phase-Contrast Tomography (XPCT) faces a fundamental physical bottleneck: ancient carbon-based soot ink ($Z = 6$) and pyrolyzed papyrus cellulose ($Z = 6$) exhibit nearly identical X-ray absorption coefficients. Standard 3D convolutional and spatiotemporal transformer models trained on raw voxel intensities suffer from two critical failure modes: (1) high false-positive rates along dense, unidirectional vascular papyrus fiber bundles, and (2) severe performance collapse under cross-scroll domain shifts when transitioning from $54\text{ keV}$ ($7.91\ \mu\text{m}$) detached fragments to intact scrolls scanned at ESRF BM18 under high-resolution ($2.399\ \mu\text{m}$, $0.22\text{ m}$, $78\text{ keV}$) or long-propagation ($8.64\text{--}9.36\ \mu\text{m}$, $1.2\text{ m}$, $113\text{--}116\text{ keV}$) synchrotron regimes.

In this paper, we introduce **Kintsugi Physical XPCT** (`KintsugiPhysicalUNet3DTo2D`), a physics-informed 3D-to-2D neural architecture that replaces raw intensity reliance with explicit differential phase-contrast and micro-morphological operators. First, we formulate a mask-normalized whole-slab 3D synchrotron calibration (`normalize_slab_globally`) that removes $>1.2\text{ mm}$ planar beam drift without canceling $0.5\text{ mm}$ calligraphic stroke interiors or creating boundary halos on rotated segments. Second, we design an 8-lens `PhysicalStem3D` incorporating fine ($d=1$) and dilated ($d=2$) 5-tap noise-robust Pavel Holoborodko axial derivatives ($\partial _ z$) and axial Laplacians ($\partial _ {zz}^2$) to capture the Fresnel phase-contrast interference triplet (central epidermal peak flanked by dark phase fringes), paired with planar thermal micro-crackle residuals ($\mathcal{R} _ {xy}$) and $2.4\ \mu\text{m}$ intra-cell crackle variance ($\sigma _ {8\times 8}$). Third, we introduce `SoftSheetUnwarper3D`, a differentiable axial alignment module that estimates a continuous vertical offset field $\Delta\hat{z}(y,x) \in [-2, +2]$ slices to flatten residual mesh undulations prior to tri-axial $(Z,Y,X)$ sequence mixing and multi-scale calligraphic context aggregation (`CalligraphicPyramid2D`). Fourth, we propose `AnisotropicFiberSuppressionLoss`, a regularization objective derived from the eigenvalues $(\lambda _ 1, \lambda _ 2)$ of the 2D papyrus structure tensor that penalizes ink predictions aligned with unidirectional vascular fibers.

Across strictly held-out benchmarks with zero spatial leakage, Kintsugi Physical XPCT: (i) improves zero-shot cross-fragment generalization (`Fragment 1` $\to$ `Fragment 3`) from `0.7511` to **`0.7805` ROC AUC** (`+12.8%` F1) while reducing fiber false positives by **`-29.7%`**; (ii) lifts held-out accuracy on Scroll 4 (`PHerc1667 w013` at $2.399\ \mu\text{m}$) from `0.5082` (zero-shot) to **`0.6740` ROC AUC** (`Best F1 = 0.5345`); (iii) lifts held-out accuracy on long-propagation scroll `PHerc0139 w044` ($9.362\ \mu\text{m}, 1.2\text{ m}, 113\text{ keV}$) from `0.5570` to **`0.7274` ROC AUC** (`AP = 0.6233`, `Best F1 = 0.6153`); and (iv) produces the first full-area ink discovery atlas across all 6 unopened `auto_grown` segments of virgin scroll **`PHerc0800`** ($8.64\ \mu\text{m}, 1.2\text{ m}, 116\text{ keV}$, $q _ {99} = 0.887\text{--}0.927$).

---

## 1. Introduction & The Cross-Scroll Domain Shift Problem

The eruption of Mount Vesuvius in 79 AD carbonized hundreds of papyrus scrolls in the Villa of the Papyri at Herculaneum. Pioneering work by **W. Brent Seales** and colleagues (Seales et al., 2016; Parker et al., 2019; Parsons et al., 2023) established the virtual unwrapping pipeline—comprising volumetric X-ray tomography, 3D papyrus sheet segmentation, surface-volume flattening, and machine-learning ink detection—culminating in the **Vesuvius Challenge** (Seales et al., 2023) and the 2023 Grand Prize reading of `PHerc.Paris.4` (Scroll 1) by **Youssef Nader, Luke Farritor, and Julian Schilliger**.

Despite this triumph on Scroll 1 (scanned at $7.91\ \mu\text{m}$, $54\text{ keV}$ at Diamond Light Source), extending automated ink detection to the broader Herculaneum library has encountered a severe **cross-scroll domain shift**. New intact scroll acquisitions performed at the **European Synchrotron Radiation Facility (ESRF) BM18 beamline** (led by **Paul Tafforeau** and colleagues) operate in two distinct physical regimes:
1. **High-Resolution Short-Propagation Regime (`PHerc1667` / Scroll 4)**: $2.399\ \mu\text{m}$ voxel size, $0.22\text{ m}$ propagation distance, $78\text{ keV}$ beam energy. Here, a typical $3.0\text{ mm}$ Greek letter spans $\sim 1{,}250\text{ px}$ (nearly $10\times$ larger than a standard $128\times 128$ receptive field).
2. **Long-Propagation High-Energy Phase-Contrast Regime (`PHerc0139`, `PHerc0800`)**: $8.64\text{--}9.362\ \mu\text{m}$ voxel size, $1.2\text{ m}$ propagation distance, $113\text{--}116\text{ keV}$ beam energy. At $113\text{--}116\text{ keV}$ and $1.2\text{ m}$ free-space propagation, pure absorption contrast between carbon soot ink ($Z=6$) and carbonized papyrus ($Z=6$) vanishes; instead, contrast is governed by **Fresnel propagation phase interference** (Mocella et al., 2015; Brun et al., 2016), producing a characteristic second-derivative axial triplet: a bright central peak at the ink epidermis flanked by negative dark interference fringes in the adjacent air/papyrus transition layers.

When standard 3D ink detectors (such as 3D ResNet, I3D, or TimeSformer models trained on raw voxel brightness $V(z,y,x)$ with small-patch local $Z$-score normalization) are evaluated across these regimes, we observe two systematic failure modes:
- **Failure Mode A (Small-Patch Normalization Stroke Cancellation & Boundary Halos)**: Computing local $Z$-scores over $64\times 64$ or $128\times 128$ patches cancels the interior of $0.5\text{ mm}$ calligraphic strokes whenever a patch falls inside a thick stroke, while zero-padded rotated diamond segments (such as `PHerc0800`) induce bright perimeter halos.
- **Failure Mode B (Vascular Fiber Confusion)**: Dense papyrus vascular bundles share the same raw brightness as carbon ink crusts, generating nearly $30\%$ false-positive activations along straight horizontal and vertical plant fibers.

---

## 2. Related Work & Scientific Foundations

Our formulation builds directly upon four foundational pillars of Herculaneum synchrotron physics and computer vision:
1. **Propagation X-Ray Phase-Contrast Tomography (XPCT) of Herculaneum Papyri**: **Vito Mocella et al. (2015)** and **Emmanuel Brun et al. (2016)** first demonstrated that carbonized Herculaneum ink can be distinguished from papyrus fibers via propagation-based phase-contrast micro-CT, while **Pieter Tack et al. (2016)** characterized the trace elemental and micro-morphological profile of the ink crust. **Paul Tafforeau et al. (ESRF BM18)** scaled multi-resolution propagation XPCT ($78\text{--}116\text{ keV}$, $0.22\text{--}1.2\text{ m}$) to intact scrolls.
2. **Virtual Unwrapping & Volumetric Ink Detection (`ink-id`)**: **W. Brent Seales, Stephen Parsons, Seth Parker, Christy Chapman**, and the EduceLab / Vesuvius Challenge team (**Giorgio Angelotti, Ben Kyles, Henderson**, et al.) created the `surface-volume` representation and the open-source `villa` / `ink-id` ecosystem (Parsons et al., 2023).
3. **2023 Vesuvius Challenge Grand Prize Architectures**: **Youssef Nader, Luke Farritor, and Julian Schilliger (2023)** demonstrated the power of 3D spatiotemporal encoders (TimeSformer / 3D ResNet ensembles) and automated segmentation (` ThaumatoAnakalyptor`) on Scroll 1.
4. **Smooth Noise-Robust Numerical Differentiation**: **Pavel Holoborodko (2008)** derived smooth low-pass digital differentiation filters that suppress high-frequency quantum and ring noise when computing first and second axial derivatives ($\partial _ z, \partial _ {zz}^2$) on discrete tomographic grids.

---

## 3. Mathematical Formulation of Kintsugi Physical XPCT

### 3.1. Mask-Normalized Global 3D Synchrotron Calibration (`normalize_slab_globally`)

Given a 3D surface-volume slab $V(z,y,x) \in \mathbb{R}^{D \times H \times W}$ and valid papyrus domain mask $M(y,x) = \mathbb{I}\!\left[\max _ z V(z,y,x) > 0\right]$, we first canonically orient the depth axis $z \in \{0,\dots,D-1\}$ so that air precedes the papyrus epidermis ($\mu _ {z < D/2} \le \mu _ {z \ge D/2}$). We then compute the 3D global mean $\mu _ {3\text{D}}$ and standard deviation $\sigma _ {3\text{D}}$ strictly over valid voxels $M(y,x)=1$, followed by mask-normalized $129\times 129$ planar drift removal:

$$\tilde{V} _ {3\text{D}}(z,y,x) = \frac{V(z,y,x) - \mu _ {3\text{D}}}{\sigma _ {3\text{D}} + \varepsilon} \cdot M(y,x)$$

$$\mathcal{D} _ {129}(y,x) = \frac{\mathrm{AvgPool} _ {129\times 129}\!\left(\bar{V} _ {xy}\cdot M\right)}{\mathrm{AvgPool} _ {129\times 129}(M) + \varepsilon}, \qquad \tilde{V} _ {\text{cal}}(z,y,x) = \left(\tilde{V} _ {3\text{D}}(z,y,x) - \gamma\,\mathcal{D} _ {129}(y,x)\right)\cdot M _ {\text{eroded}}(y,x)$$

where $\gamma = 0.7$. Because the $129\times 129$ kernel ($\sim 2.4\text{ mm}$ at $18.7\ \mu\text{m}/\text{px}$) is nearly $5\times$ wider than a $0.5\text{ mm}$ Greek calligraphic stroke, $\mathcal{D} _ {129}$ removes low-frequency synchrotron beam inhomogeneity while preserving stroke interiors and eliminating rotated-diamond perimeter halos.

### 3.2. Multi-Scale 8-Lens Differential Physical Stem (`PhysicalStem3D`)

Empirical profiling of human-labeled ink vs. bare papyrus on `PHerc0139` ($9.362\ \mu\text{m}, 1.2\text{ m}, 113\text{ keV}$) reveals that ink voxels exhibit a $+19.94$ intensity peak at the sheet epidermis ($z = 11\text{--}14$) flanked by negative Fresnel dark phase fringes of $-10.03$ at $z = 8$ and $-11.21$ at $z = 16$. To capture both thin ($d=1$) and dilated ($d=2$) phase fringes alongside epidermal micro-crackle, `PhysicalStem3D` constructs an 8-channel physical tensor $\mathbf{\Phi}(V) \in \mathbb{R}^{8 \times D \times H \times W}$:

$$\partial _ z^{(d)}\tilde{V} = \tilde{V} * _ d \left(k _ z^{(1)} \otimes g _ 3 \otimes g _ 3\right), \qquad k _ z^{(1)} = \frac{1}{8}[-1,\,-2,\,0,\,+2,\,+1]$$

$$\partial _ {zz}^{(d)}\tilde{V} = \tilde{V} * _ d \left(k _ z^{(2)} \otimes g _ 3 \otimes g _ 3\right), \qquad k _ z^{(2)} = \frac{1}{4}[+1,\,0,\,-2,\,0,\,+1]$$

$$\mathcal{R} _ {\text{crackle}}(z,y,x) = \tilde{V}(z,y,x) - (\tilde{V} * G _ {5\times 5})(z,y,x) + \alpha _ {\text{uf}}\,\sigma _ {8\times 8}\!\left(V _ {2.4\mu\text{m}}\right)$$

where $d \in \{1, 2\}$ denotes the axial dilation factor and $\sigma _ {8\times 8}(V _ {2.4\mu\text{m}})$ preserves sub-cell $2.399\ \mu\text{m}$ thermal mud-cracking variance when pooling Scroll 4 to the unified calligraphic scale ($19.19\ \mu\text{m}/\text{px}$).

### 3.3. Differentiable 3D Soft Sheet Unwarping (`SoftSheetUnwarper3D`) & `CalligraphicPyramid2D`

Even after geometric flattening (`flatboi` / `auto_grown`), papyrus fibers undulate across $\pm 2$ depth slices. `SoftSheetUnwarper3D` predicts a per-column axial attention distribution $w _ z(y,x) = \mathrm{softmax} _ z(s _ z(y,x))$ and estimates a smooth vertical displacement field:

$$\Delta\hat{z}(y,x) = \Delta z _ {\max} \cdot \tanh\!\left(\frac{\mathcal{A} _ {3\times 3}\!\left(\sum _ {z=0}^{D-1}(z - z _ c)\,w _ z(y,x)\right)}{\Delta z _ {\max}}\right), \qquad \Delta z _ {\max} = 2.0$$

The feature volume is differentiably resampled along $Z$ at $z' = z + \Delta\hat{z}(y,x)$, processed through a tri-axial bidirectional sequence mixer (`AxialSequenceMixer3D`), projected to 2D via `LearnedDepthHead`, and passed through `CalligraphicPyramid2D` ($1\times \to \tfrac{1}{2}\times \to \tfrac{1}{4}\times \to \tfrac{1}{8}\times$ with dilated bottleneck) so the 2D receptive field encompasses full $3\text{ mm}$ Greek letters.

### 3.4. Structure-Tensor Anisotropic Fiber-Texture Suppression Loss

To prevent the network from firing on straight papyrus vascular striations, we compute the 2D structure tensor $J(y,x) = G _ \rho * (\nabla S\,\nabla S^\top)$ of the mean papyrus surface $S(y,x) = \frac{1}{D}\sum _ z V(z,y,x)$. From the eigenvalues $\lambda _ 1 \ge \lambda _ 2 \ge 0$ of $J(y,x)$, the local fiber anisotropy coherence is:

$$C _ {\text{fiber}}(y,x) = \left(\frac{\lambda _ 1(y,x) - \lambda _ 2(y,x)}{\lambda _ 1(y,x) + \lambda _ 2(y,x) + \varepsilon}\right)^2 \in [0, 1]$$

Unidirectional papyrus fibers satisfy $\lambda _ 1 \gg \lambda _ 2$ ($C _ {\text{fiber}} \to 1$), whereas curved calligraphic strokes and junctions are multi-directional ($C _ {\text{fiber}} \ll 1$). Our total training objective is:

$$\mathcal{L} _ {\text{total}} = \mathcal{L} _ {\text{Fable}}(\mathrm{BCE} + \mathrm{Tversky}) + \lambda _ {\text{fiber}}\frac{\sum _ {y,x} C _ {\text{fiber}}(y,x)\,p(y,x)^2\,(1 - Y(y,x))}{\sum _ {y,x}(1 - Y(y,x))} + \lambda _ {\text{TV}}\Vert{}\nabla _ {xy}\Delta\hat{z}\Vert{} _ 1$$

---

## 4. Experimental Results (Strictly Held-Out Evaluation)

### 4.1. Cross-Fragment Generalization (`Fragment 1` $\to$ Held-Out `Fragment 3`)

| Architecture | Held-Out ROC AUC | Average Precision (AP) | Best F1 Score | False Positives ($p > 0.5$) |
| :--- | :---: | :---: | :---: | :---: |
| Residual 3D U-Net (`residual_unet`) | `0.7511` | `0.2194` | `0.3016` | `39,586 px` (Baseline) |
| Axial 3D Mixer (`kintsugi_axial`, no physics) | `0.7330` | `0.1945` | `0.3046` | `40,287 px` (`+1.8%`) |
| **Kintsugi Physical (`kintsugi_physical` + 8-fold TTA)** | **`0.7805` (`+3.91%`)** | **`0.2323` (`+5.88%`)** | **`0.3401` (`+12.8%`)** | **`27,825 px` (`-29.7%` FP)** |

### 4.2. Intact Full-Scroll Domain Adaptation (`PHerc1667 w013` & `PHerc0139 w044`)

| Scroll Regime & Synchrotron Parameters | Training Split | Strictly Held-Out Validation Split | Zero-Shot Fragment AUC | **Scroll-Adapted Kintsugi ROC AUC** | **Held-Out AP / Best F1** |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Scroll 4 (`PHerc1667`)** ($2.399\ \mu\text{m}, 0.22\text{ m}, 78\text{ keV}$) | `w018` + disjoint `w013` (`cy=20..40`, `cy=150..170`) | **`PHerc1667 w013`** (`cy=86..106, cx=62..87`) | `0.5082` | **`0.6740` (`0.6459` @ `bc=24`)** | **AP `0.5045` / F1 `0.5345`** |
| **Scroll `PHerc0139`** ($9.362\ \mu\text{m}, 1.2\text{ m}, 113\text{ keV}$) | `w035`, `w040`, `w041` | **`PHerc0139 w044`** (`cy=28..44, cx=20..36`) | `0.5570` | **`0.7274` (`+0.1704`)** | **AP `0.6233` / F1 `0.6153`** |

### 4.3. First Ink Discovery Atlas on Virgin Scroll `PHerc0800` ($8.64\ \mu\text{m}, 1.2\text{ m}, 116\text{ keV}$)

Applying our dual-domain ensemble (`model_coarse` trained on the physical twin `PHerc0139` + `model_2p4` trained on `PHerc1667`) across all 6 unopened `auto_grown` segments of `PHerc0800` (`level=1`, $8.66\text{ cm}^2$ valid papyrus area, `0` prior `ink-detection` maps in S3) reveals horizontal bands of $3\text{ mm}$-scale calligraphic stroke candidates across the 5 main segments ($q _ {99} = 0.887\text{--}0.927$).

---

## 5. References

1. **Seales, W. B.**, Parker, C. S., Segal, M., Tov, E., Shor, P., & Porath, Y. (2016). *From damage to discovery via virtual unwrapping: Reading the scroll from En-Gedi*. Science Advances, 2(9), e1601247.
2. **Mocella, V.**, Brun, E., Ferrero, C., & Delattre, D. (2015). *Revealing letters in rolled Herculaneum papyri by X-ray phase-contrast imaging*. Nature Communications, 6, 5895.
3. **Brun, E.**, Cotte, M., Wright, J., Ruat, M., Tack, P., Vincze, L., Ferrero, C., Delattre, D., & Mocella, V. (2016). *Revealing metallic ink in Herculaneum papyri*. Proceedings of the National Academy of Sciences (PNAS), 113(14), 3751–3754.
4. **Tack, P.**, Cotte, M., Bauters, S., Brun, E., Banerjee, D., Bras, W., Ferrero, C., Delattre, D., Mocella, V., & Vincze, L. (2016). *Tracking ink composition on Herculaneum papyrus scrolls: quantification and speciation of lead by X-ray based techniques and Monte Carlo simulations*. Scientific Reports, 6, 20763.
5. **Parsons, S.**, Parker, C. S., Chapman, C., Hayashida, M., & **Seales, W. B.** (2023). *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using Controlled-Sensitivity Phase-Contrast X-Ray Tomography and 3D Convolutional Neural Networks (`ink-id`)*. Digital Humanities / University of Kentucky.
6. **Seales, W. B.**, **Friedman, N.**, **Gross, D.**, **Parsons, S.**, **Angelotti, G.**, **Kyles, B.**, & **Tafforeau, P.** (2023–2026). *The Vesuvius Challenge: Reading the Herculaneum Papyri with Particle Accelerators and Machine Learning*. [https://scrollprize.org](https://scrollprize.org).
7. **Nader, Y.**, **Farritor, L.**, & **Schilliger, J.** (2023). *Vesuvius Challenge 2023 Grand Prize Winning Solution: 3D TimeSformer & ResNet Ensembles for Volumetric Ink Detection on PHerc.Paris.4*.
8. **Holoborodko, P.** (2008). *Smooth Noise Robust Differentiators*. [http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/](http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/).
