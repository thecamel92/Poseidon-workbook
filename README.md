# Poseidon (scOT): a guided walkthrough

Two notebooks investigating Poseidon-L, a foundation model for PDEs, on compressible
Euler and incompressible Navier-Stokes problems.

| file | what it is |
|---|---|
| `poseidon_exercises.ipynb` | the walkthrough with the key blocks left blank to fill in |
| `poseidon_solutions.ipynb` | the same notebook, complete |

The two are identical apart from those blanks, so they can be read side by side.

## Running it in Google Colab

**1. Open the notebook.** `File -> Open notebook -> GitHub`, paste this repository's URL,
and pick `poseidon_exercises.ipynb`. Colab loads the notebook only, not the repository.

**2. Turn on the GPU.** `Runtime -> Change runtime type -> T4 GPU -> Save`. The notebook
picks the fastest backend it can find, so there is nothing to edit.

**3. Install the one pinned dependency, then restart.**

```python
!pip install -q "transformers<5"
```

Then `Runtime -> Restart session`. **This restart is required** - Colab has already
imported a newer transformers, and only a restart makes the downgrade take effect.
Restarting does not delete downloaded files.

*Why:* transformers 5.0 removed the `head_mask` argument from `Swinv2Attention.forward`,
which scOT calls positionally. Any 4.x release works. Skipping this step produces a
`TypeError` deep inside the attention layer the first time the model loads.

**4. Clone this repository and move into it.**

```python
!git clone https://github.com/<owner>/<repo>.git
%cd <repo>
```

This brings down about 4 MB: the notebooks, `scOT/` and `finitevolume_python/`. Section 1
resolves every path from the working directory, so the `%cd` is what makes them line up.

**5. Run the notebook from the top.** The first cell of section 1 downloads the weights and
the data, then loads the model. Expect it to take a few minutes; nothing else in the
notebook is slow.

**If the session drops**, re-run steps 4 and 5. The download skips whatever is already
there, so re-running is cheap and safe.

Nothing needs cleaning up afterwards. Colab destroys the machine and everything with it.

## Running it locally

Open either notebook **from inside this folder** and run the cells in order. Section 1
resolves paths relative to the working directory, so the folder can be moved or renamed,
but the notebook has to be opened from within it.

Put the files below in place yourself, or let section 1 download them.

## The weights and the data

Not in this repository - about 17 GB in total. Section 1 downloads them from the
[CAMLab collections](https://huggingface.co/camlab-ethz) on the Hugging Face Hub into
`weights/` and `data/`, which are empty here on purpose.

| lands at | from | size |
|---|---|---|
| `weights/Poseidon-L/` | `camlab-ethz/Poseidon-L` | 2.5 GB |
| `data/CE-KH/CE-KH.nc` | `camlab-ethz/CE-KH`, `data_0.nc` | 4.9 GB |
| `data/CE-Gauss/CE-Gauss.nc` | `camlab-ethz/CE-Gauss`, `data_0.nc` | 4.9 GB |
| `data/NS-SL/NS-SL.nc` | `camlab-ethz/NS-SL`, `velocity_0.nc` | 4.8 GB |

Each dataset is published as a dozen or more interchangeable shards, and one is enough:
the notebook only ever reads the held-out test split. **The shards are renamed on the way
in**, because scOT hardcodes the filename it opens for each dataset - `NS-SL` in particular
ships as `velocity_0.nc`.

If a download crawls at around 10 kB/s, the Xet transfer backend is the cause. Section 1
already sets `HF_HUB_DISABLE_XET=1` to force the classic CDN path, which runs at
10-40 MB/s.

## What is in here

```
poseidon_solutions.ipynb     the complete walkthrough
poseidon_exercises.ipynb     the same, with blanks to fill in
scOT/                        the model and its dataset readers, from camlab-ethz/poseidon
finitevolume_python/         the finite-volume Euler solver used in section 5
data/                        empty; the datasets land here
weights/                     empty; the weights land here
```

`scOT/` and `finitevolume_python/` are vendored rather than installed. Section 1 puts this
folder on `sys.path`, so neither needs a `pip install`.

## Requirements

Python 3.10 with `torch`, `transformers<5`, `accelerate`, `numpy`, `matplotlib`, `h5py`
and `huggingface_hub`. Only the transformers bound is strict; the rest are ordinary
recent versions, and Colab already has them.

On Apple silicon the notebook sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch.
Poseidon is a Swin transformer whose shifted-window attention calls `torch.roll`, which has
no MPS kernel - without the flag the first forward pass raises.
