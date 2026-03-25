# NeuTTS Air dLLM

NeuTTS Air model which is further trained unofficially for block-causal generation capability.

All the credits and license for the amazing base NeuTTS model from [NeuTTS-Air](https://huggingface.co/neuphonic/neutts-air).

Modeling and configuration implementation is heavily borrowed from [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), so it can be easily train based on their codes if you want.

For brief summary of what is done, refer to my [post](https://balladie.github.io/blog/2026/bdtts/).

## Installation

Tested on Python 3.12.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python generate.py \
    --text "Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, intending to safeguard some from the harsh truths." \
    --ref_audio ./samples/dave.wav \
    --ref_text ./samples/dave.txt 
```