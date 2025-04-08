# MusicDiffGen : Binomial Diffusion for Symbolic Music Generation

This project demonstrates a **binomial diffusion** approach for generating symbolic music (piano rolls) using the **MAESTRO v3.0.0** dataset. The main code is implemented in a Jupyter notebook, walking through:

1. GPU setup (selecting a specific GPU by ID).  
2. **Data Preprocessing** from MAESTRO MIDI files into piano roll segments.  
3. A custom **PyTorch Dataset** (`MaestroBinomialDiffusionDataset`) for training.  
4. A **UNet-like model** (`MusicDiffusionGenerator`) for binomial diffusion.  
5. **Training** and **Sampling** the model, including optional **partial infilling**.

---

## Method Overview

### 1. Data Preprocessing (MAESTRO Dataset)

- **Dataset**: [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) contains ~200 hours of virtuosic piano performances with aligned MIDI and audio.  
- **Goal**: Convert MIDI files into binary piano roll matrices of shape  
  $$
  (T,\ \text{pitch\_range})
  $$  
  suitable for binomial diffusion.
- **Steps**:
  1. Read `maestro-v3.0.0.csv`, filtering by split (train/validation/test).  
  2. Load each MIDI file using [pretty_midi](https://github.com/craffel/pretty-midi).  
  3. Restrict pitch range to `[21...108]`, binarize notes, and segment in time.  
  4. Concatenate valid segments into a list of numpy arrays.

### 2. MaestroBinomialDiffusionDataset

- For each segment, returns a tuple $(x_\mathrm{noisy}, x_0, t)$ where:
  - $x_0$ is the clean binary roll.
  - $t \in [0, \text{num\_steps}-1]$ is the diffusion timestep.
  - $x_\mathrm{noisy}$ is generated via:

    $$
    p(\text{noisy}[i,j]) = x_0[i,j] \cdot (1 - \beta_t) + \text{ratio} \cdot \beta_t
    \quad \text{with} \quad \beta_t = \frac{t+1}{\text{num\_steps}}
    $$

- If not specified, `ratio` is computed from the dataset (mean density of 1s).

### 3. UNet-Like Model (MusicDiffusionGenerator)

- A simplified **UNet** architecture adapted to 2D data with shape `(Time, Pitch)`.  
- Downsampling is applied only on the time axis (not pitch).  
- A small MLP embeds the diffusion step $t$, injected into each UNet block.  
- The model predicts $\hat{x}_0$, the denoised piano roll.

### 4. Training

- **Loss**: L1 loss between $\hat{x}_0$ and $x_0$.  
- **Pipeline**:
  - Batch input: $(x_\text{noisy}, x_0, t)$  
  - Optimizer: Adam  
  - Typical learning rate: $1 \times 10^{-4}$ or $5 \times 10^{-5}$  
  - Train for 5–50 epochs

### 5. Sampling

- Starts from random binomial noise with probability = `ratio`.
- At each step:
  1. Predict $\hat{x}_0$
  2. Binarize it with threshold 0.5
  3. Apply a stochastic XOR mask with the original noise $x_T$, scaled by $\beta_t$
  4. Iterate backward from $t = T$ to $t = 0$

- Output is a binary matrix representing a piano roll.
- Optional playback using `pretty_midi + pyfluidsynth` (SoundFont required).

---

## Key Results

- The model can generate coherent 16-beat piano roll segments.
- Supports:
  - **Partial generation** from a prompt (e.g., conditioning on the first half).
  - **Melody harmonization** by freezing certain pitches and diffusing others.

---

## Installation

Required Python packages:

- `torch`, `torchvision`, `torchaudio`
- `pretty_midi`, `pandas`, `numpy`, `scipy`
- `tqdm`, `pyfluidsynth` (optional, for audio rendering)
- `matplotlib` (for visualization)

Install with:

```bash
pip install torch torchvision torchaudio tqdm pretty_midi pandas pyfluidsynth scipy matplotlib
```

## How to Run

### 1. Clone the Repository

Clone this repository and keep all scripts in the same folder.

### 2. Download MAESTRO v3.0.0

- Place the folder `maestro-v3.0.0/` in the root directory.
- It should contain subfolders like `2004/` and the CSV file `maestro-v3.0.0.csv`.

### 3. Set GPU Device

- Use `gpu_id = 0` or `1`, as available.
- Confirm that:

  ```python
  torch.cuda.is_available() is True

### 4. Preprocess the Data

- Run the following function with appropriate parameters:

  ```python
  build_maestro_segments(
      maestro_csv="maestro-v3.0.0/maestro-v3.0.0.csv",
      split="train",
      pitch_low=21,
      pitch_high=108,
      resolution=24,
      segment_beats=16,
      base_dir="maestro-v3.0.0"
  )
  ```

- Construct the dataset using:

  ```python
  MaestroBinomialDiffusionDataset(segments_list, num_steps=100, ratio=0.03)
  ```

---

### 5. Train the Model

- Call the training function with your dataset:

  ```python
  train_diffusion_model(
      model,
      dataset,
      batch_size=256,
      lr=1e-4,
      epochs=5,
      device="cuda"
  )
  ```

- The training loop will log the **L1 loss** at each step.

---

### 6. Generate Samples

- Use one of the following functions to generate and play music:

  ```python
  generate_and_play_audio_from_model(
      model=model,
      out_name="sample_01",
      shape=(384, 88),
      pitch_low=21,
      ratio=0.3,
      total_steps=100,
      device="cuda",
      out_dir="experiments",
      sf2_path="soundfonts/FluidR3_GM.sf2"
  )
  ```

  Or to generate only the roll:

  ```python
  sample_binomial_diffusion(
      model=model,
      shape=(384, 88),
      total_steps=100,
      ratio=0.3,
      device="cuda"
  )
  ```

- Output includes:
  - `.mid` file — generated MIDI roll
  - `.wav` file — synthesized audio (if `pyfluidsynth` is configured)

---

## Notes

- **Memory Usage**  
  Use smaller `batch_size` if you encounter out-of-memory (OOM) errors.

- **Masking / Inpainting**  
  You can fix specific time/pitch regions during sampling (e.g., freeze melody and generate harmony).

- **SoundFont**  
  A `.sf2` SoundFont file is required for audio synthesis using `pyfluidsynth`. Example:

  ```bash
  soundfonts/FluidR3_GM.sf2
  ```

---

## Acknowledgments

- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) by Magenta  
- [pretty_midi](https://github.com/craffel/pretty-midi)  
- Diffusion model inspired by [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
