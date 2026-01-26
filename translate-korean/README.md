# Korean Number Translation Experiment

This experiment trains a character-level language model to learn Korean number translations.

## Data Format

The model learns from a stream of Korean-English number pairs:
- 일 one
- 이 two
- 삼 three
- 사 four
- 오 five
- ...

## Directory Structure

```
translate-korean/
├── README.md
├── config/
│   └── basic.py
├── data/
│   └── basic/
│       ├── prepare.py
│       ├── meta.pkl
│       ├── train.bin
│       └── val.bin
└── out/
```

## Setup

1. Create Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install torch numpy tqdm
   ```

3. Prepare data:
   ```powershell
   cd translate-korean
   python data/basic/prepare.py
   ```

## Training

From the `translate-korean` directory:

```powershell
$env:NANOGPT_CONFIG = "..\comp560-nanoGPT\configurator.py"
python -u ..\comp560-nanoGPT\train.py config/basic.py
```

## Sampling

After training:

```powershell
$env:NANOGPT_CONFIG = "..\comp560-nanoGPT\configurator.py"
python -u ..\comp560-nanoGPT\sample.py config/basic.py --num_samples=1 --max_new_tokens=100 --seed=2345
```

## Experiment Log

### Run 1: Debug (max_iters=200)
- Purpose: Verify workflow
- Results: [To be filled after running]

### Run 2: Main (max_iters=2000)
- Purpose: Actual training
- Results: [To be filled after running]
