#!/usr/bin/env python3
"""
NeuTTS Air - Single Generation Script
Modified by Gangin Park for dLLM Generation

Usage:
    python generate.py --text "Hello world" --ref_audio path/to/audio.wav --ref_text "Reference text" --output output.wav
"""

import argparse
from pathlib import Path
import librosa
import numpy as np
import torch
import re
import os
import soundfile as sf
import time
from neucodec import NeuCodec, DistillNeuCodec
from phonemizer.backend import EspeakBackend


class NeuTTSGenerator:
    def __init__(
        self,
        backbone_repo="ssonpull519/neutts-air-dllm-bd8",
        backbone_device="cuda",
        lora_repo=None,
        codec_repo="neuphonic/neucodec",
        codec_device="cuda",
    ):
        self.sample_rate = 24_000
        self.max_context = 2048
        
        # Flags
        self._is_quantized_model = False
        self._is_onnx_codec = False
        self.tokenizer = None
        
        print("Loading phonemizer...")
        self.phonemizer = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        
        self._load_backbone(backbone_repo, backbone_device, lora_repo)
        self._load_codec(codec_repo, codec_device)
        
        # Optional: Load watermarker
        try:
            import perth
            self.watermarker = perth.PerthImplicitWatermarker()
        except ImportError:
            print("Warning: perth watermarker not available, skipping watermarking")
            self.watermarker = None

    def _load_backbone(self, backbone_repo, backbone_device, lora_repo):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device}...")
        print(f"Loading lora from: {lora_repo}...")
        
        if backbone_repo.endswith("gguf"):
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with: pip install llama-cpp-python"
                ) from e
            
            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device == "gpu" else False,
            )
            self._is_quantized_model = True
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo, trust_remote_code=True)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo, trust_remote_code=True).to(
                torch.device(backbone_device)
            )
            if lora_repo:
                self.backbone.load_adapter(lora_repo)

    def _load_codec(self, codec_repo, codec_device):
        print(f"Loading codec from: {codec_repo} on {codec_device}...")
        
        if "neucodec-onnx-decoder" in codec_repo:
            if codec_device != "cpu":
                raise ValueError("Onnx decoder only currently runs on CPU.")
            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import the onnx decoder. "
                    "Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                ) from e
            self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
            self._is_onnx_codec = True
        elif "distill-neucodec" in codec_repo:
            self.codec = DistillNeuCodec.from_pretrained(codec_repo)
            self.codec.eval().to(codec_device)
        elif "neucodec" in codec_repo:
            self.codec = NeuCodec.from_pretrained(codec_repo)
            self.codec.eval().to(codec_device)
        else:
            raise ValueError(
                "Invalid codec repo! Must be one of: "
                "'neuphonic/neucodec', 'neuphonic/distill-neucodec', "
                "'neuphonic/neucodec-onnx-decoder'."
            )

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        return " ".join(phones)

    def encode_reference(self, ref_audio_path: str | Path) -> np.ndarray:
        """Encode reference audio to codes."""
        print(f"Encoding reference audio: {ref_audio_path}")
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str) -> np.ndarray:
        """Decode speech tokens to waveform."""
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]
        
        if len(speech_ids) == 0:
            raise ValueError("No valid speech tokens found in the output.")
        
        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                    self.codec.device
                )
                recon = self.codec.decode_code(codes).cpu().numpy()
        
        return recon[0, 0, :]

    def _apply_chat_template(self, ref_codes, ref_text: str, input_text: str) -> list[int]:
        """Apply chat template for torch inference."""
        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
        
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)
        
        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1:]
        )
        
        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        
        return ids

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        """Inference using torch backend."""
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        st = time.time()
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                mask_id=self.tokenizer.convert_tokens_to_ids("|<MASK>|"),
                stop_token=speech_end_id,
                block_size=8,
                max_new_tokens=self.max_context,
                small_block_size=4,
                threshold=0.0175,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                use_block_cache=False,
            )

        gen_t = time.time() - st

        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), 
            add_special_tokens=False
        )
        num_tokens = output_tokens.shape[1] - input_length

        print("Elapsed Time:", "Generation %.4f secs (%.4f tok/s)" % (gen_t, num_tokens / gen_t))
        print("Generated Length:", num_tokens, "tokens")

        return output_str

    def _infer_ggml(self, ref_codes, ref_text: str, input_text: str) -> str:
        """Inference using GGML/llama.cpp backend."""
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)
        
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        return output["choices"][0]["text"]

    def generate(
        self, 
        text: str, 
        ref_audio_path: str | Path, 
        ref_text: str | Path,
        output_path: str | Path = "tmp.wav",
        num_gen: int = 1,
    ):
        """
        Generate speech from text using reference audio.
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio file
            ref_text: Transcript of reference audio
            output_path: Where to save output audio
        """
        print(f"\nGenerating speech for: '{text}'")
        
        # Encode reference
        ref_codes = self.encode_reference(ref_audio_path)
        
        if ref_text and os.path.exists(ref_text):
            with open(ref_text, "r") as f:
                ref_text = f.read().strip()

        output_suffix = Path(output_path).suffix

        # Generate tokens
        for i in range(num_gen):
            print(f"[{i}] Generating speech tokens...")
            if self._is_quantized_model:
                output_str = self._infer_ggml(ref_codes, ref_text, text)
            else:
                prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
                output_str = self._infer_torch(prompt_ids)
            
            # Decode to audio
            print(f"[{i}] Decoding to audio...")
            st = time.time()
            wav = self._decode(output_str)
            dec_t = time.time() - st
            print("Codec Time:", "%.4f secs" % (dec_t))
            
            # Apply watermark if available
            if self.watermarker is not None:
                wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
            
            # Save
            output_path_iter = output_path.replace(output_suffix, f"_{i}" + output_suffix)
            output_path_iter = Path(output_path_iter)
            output_path_iter.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path_iter, wav, self.sample_rate)
            print(f"[{i}] ✓ Audio saved to: {output_path_iter}")
        
        # return wav


def main():
    parser = argparse.ArgumentParser(description="Generate speech with NeuTTS Air")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference audio")
    parser.add_argument("--ref_text", type=str, required=True, help="Transcript of reference audio")
    parser.add_argument("--output", type=str, default="tmp.wav", help="Output audio path")
    parser.add_argument("--backbone_repo", type=str, default="ssonpull519/neutts-air-dllm-bd8", 
                       help="Backbone model repo or path")
    parser.add_argument("--backbone_device", type=str, default="cuda", 
                       choices=["cpu", "cuda", "mps"], help="Device for backbone")
    parser.add_argument("--lora_repo", type=str, default=None, 
                       help="LoRA model repo or path")
    parser.add_argument("--codec_repo", type=str, default="neuphonic/neucodec",
                       help="Codec model repo")
    parser.add_argument("--codec_device", type=str, default="cuda",
                       choices=["cpu", "cuda", "mps"], help="Device for codec")
    parser.add_argument("--num_gen", type=int, default=1, help="Number of outputs to synthesize")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = NeuTTSGenerator(
        backbone_repo=args.backbone_repo,
        backbone_device=args.backbone_device,
        lora_repo=args.lora_repo,
        codec_repo=args.codec_repo,
        codec_device=args.codec_device,
    )
    
    # Generate
    generator.generate(
        text=args.text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        num_gen=args.num_gen,
    )


if __name__ == "__main__":
    main()
