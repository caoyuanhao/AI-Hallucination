# Baseline reproduction setup

Reproduces LettuceDetect's reported RAGTruth numbers as the baseline for this project.

## Environment gotchas (hit on this machine, worth keeping)

**1. Do not install into the Anaconda base env.**
`lettucedetect` requires `torch>=2.6.0` and `numpy>=2.2.2`. The Anaconda base env here
had `numpy 1.26.4`; installing would force-upgrade it to 2.x and break packages pinned
to numpy<2 (scikit-image, etc). Use a dedicated venv.

**2. `pip install torch` installs the wrong CUDA build.**
The GPU on this machine is a GTX 1650 with driver **462.30** (2021). That driver only
supports CUDA 11.x — CUDA 12.x needs driver >= 527.41 on Windows. The default PyPI
torch wheel is a cu12x build, so `torch.cuda.is_available()` returns False (or CUDA
fails to init). Install the **cu118** build explicitly from the PyTorch index.

`torch 2.7.1+cu118` is the newest cu118 wheel for cp312/win_amd64 and satisfies
lettucedetect's `torch>=2.6.0`.

## Steps

```bash
# 1. venv (Python 3.12)
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip

# 2. torch, cu118 build (see gotcha 2)
.venv/Scripts/python.exe -m pip install torch==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. lettucedetect (pulls transformers, numpy 2.x, etc — leaves torch alone)
.venv/Scripts/python.exe -m pip install lettucedetect

# 4. upstream repo, for its preprocess + evaluate scripts (not on PyPI)
git clone --depth 1 https://github.com/KRLabsOrg/LettuceDetect.git vendor/LettuceDetect

# 5. RAGTruth dataset
mkdir -p data/ragtruth
curl -sL -o data/ragtruth/response.jsonl \
    https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl
curl -sL -o data/ragtruth/source_info.jsonl \
    https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl

# 6. preprocess into lettucedetect's schema -> data/ragtruth/ragtruth_data.json
.venv/Scripts/python.exe vendor/LettuceDetect/lettucedetect/preprocess/preprocess_ragtruth.py \
    --input_dir data/ragtruth --output_dir data/ragtruth
```

## Verify the data before trusting any number

```
total samples          17790
split                  train 15090 / test 2700
test task types        Summary 900 / Data2txt 900 / QA 900
test hallucinated      943 (34.9%)
```

These match the RAGTruth paper. If they don't, the preprocessing silently dropped rows.

## Run the evaluation

```bash
# batch_size tuned for 4GB VRAM: 4 for base, 2 for large
.venv/Scripts/python.exe vendor/LettuceDetect/scripts/evaluate.py \
    --model_path KRLabsOrg/lettucedect-base-modernbert-en-v1 \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type example_level --batch_size 4 \
    > benchmarks/results/base_example_level.txt 2>&1

.venv/Scripts/python.exe benchmarks/parse_results.py benchmarks/results/*.txt --markdown
```

`--evaluation_type` accepts `example_level`, `token_level`, `char_level`.
`char_level` is the span-level metric and runs one sample at a time (slower).

## Versions used

| | |
|---|---|
| Python | 3.12.4 |
| torch | 2.7.1+cu118 |
| transformers | 5.16.1 |
| lettucedetect | 0.2.3 |
| GPU | GTX 1650, 4GB, driver 462.30 |
