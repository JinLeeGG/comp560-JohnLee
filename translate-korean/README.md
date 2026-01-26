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
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy tqdm
   ```

3. Prepare data:
   ```bash
   cd translate-korean
   python data/basic/prepare.py
   ```

## Training

From the `translate-korean` directory:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

## Sampling

After training:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=1 --max_new_tokens=100 --seed=2345
```

## Experiment Log

### Run 1: Debug (max_iters=200)
- **Purpose**: Verify workflow
- **Config**: max_iters=200, device=cpu, compile=False
- **Results**: 
  - Workflow verified successfully
  - Some translations had typos (e.g., 삼→thire, 오→sive, 십육→sighteeen)
  - Some translations were correct (e.g., 십이→twelve, 팔→eight)
  - Sample output: 삼 thire, 십이 twelve, y, 십이 two, 오 sive, 십구 nineteen, 팔 eight, 십육 sighteeen, 십육 sixteen, 육 sixt, 십삼 thirteeen, 팔

### Run 2: Main (max_iters=2000)
- **Purpose**: Actual training with more iterations
- **Config**: max_iters=2000, device=cpu (GPU attempted but encountered dt=0 error, reverted to CPU), compile=False
- **Results**: 
  - **All translations were accurate!**
  - Sample output: 오 five, 십일 eleven, 팔 eight, 사 four, 이십 twenty, 이 two, 십삼 thirteen, 삼 three, 십팔 eighteen, 십이 twelve, 십육 sixteen
  - Significant improvement from Run 1 - no typos observed
  - Model successfully learned the Korean-to-English number translation pattern

### Conclusion
- **What worked**: 
  - Increasing max_iters from 200 to 2000 dramatically improved translation accuracy
  - Character-level learning successfully captured the translation pattern
  - CPU training was sufficient for this small model
- **What didn't work**: 
  - GPU training encountered a ZeroDivisionError due to dt=0 (iterations too fast)
  - Fixed by modifying model.py line 300: `flops_achieved = flops_per_iter * (1.0/dt) if dt > 0 else 0.0`
  - Posted This issue on Teams nanoGPT channel
- **Changes made**: 
  - Increased max_iters from 200 to 2000
  - Modified comp560-nanoGPT/model.py to handle dt=0 case for GPU training
  - Used CPU for final training to avoid GPU timing issues
