import io
import os
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC


class TextToSpeech:
    def __init__(
        self,
        model_path: str,
        snac_model_name: str = "hubertsiuzdak/snac_24khz",
        device: str = None,
        sample_rate: int = 24000,
    ):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.sample_rate = sample_rate

        # Load models
        self.snac = SNAC.from_pretrained(snac_model_name).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype_map = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map.get(self.device, torch.float32)
        ).to(self.device)

        # Special tokens
        self.start_token = torch.tensor([[128259]], dtype=torch.int64)
        self.end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

        print(f"Loaded models on {self.device}")

    # -------------------------
    # Prompt handling
    # -------------------------
    def build_inputs(self, prompts, voice: str):
        prompts = [f"{voice}: {p}" for p in prompts]

        all_input_ids = [
            self.tokenizer(p, return_tensors="pt").input_ids
            for p in prompts
        ]

        modified = []
        for ids in all_input_ids:
            full = torch.cat([self.start_token, ids, self.end_tokens], dim=1)
            modified.append(full)

        return modified

    def pad_inputs(self, sequences):
        max_len = max(seq.shape[1] for seq in sequences)

        padded, masks = [], []

        for seq in sequences:
            pad_len = max_len - seq.shape[1]

            padded_seq = torch.cat(
                [torch.full((1, pad_len), 128263, dtype=torch.int64), seq],
                dim=1
            )

            mask = torch.cat(
                [
                    torch.zeros((1, pad_len), dtype=torch.int64),
                    torch.ones((1, seq.shape[1]), dtype=torch.int64),
                ],
                dim=1,
            )

            padded.append(padded_seq)
            masks.append(mask)

        return (
            torch.cat(padded, dim=0).to(self.device),
            torch.cat(masks, dim=0).to(self.device),
        )

    # -------------------------
    # Generation
    # -------------------------
    @torch.no_grad()
    def generate(self, input_ids, attention_mask):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=128258,
        )

    # -------------------------
    # Token processing
    # -------------------------
    def crop_after_token(self, generated_ids, token=128257):
        indices = (generated_ids == token).nonzero(as_tuple=True)

        if len(indices[1]) > 0:
            last_idx = indices[1][-1].item()
            return generated_ids[:, last_idx + 1 :]

        return generated_ids

    def process_rows(self, cropped_tensor):
        results = []

        for row in cropped_tensor:
            row = row[row != 128258]        # remove EOS token
            length = (row.size(0) // 7) * 7
            row = row[:length]
            row = (row - 128266).tolist()   # vectorized subtract, then move to CPU once
            results.append(row)

        return results

    # -------------------------
    # Audio decoding
    # -------------------------
    def redistribute_codes(self, code_list):
        layer_1, layer_2, layer_3 = [], [], []

        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))

        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
        ]

        return self.snac.decode(codes)

    def decode_all(self, code_lists):
        return [self.redistribute_codes(c) for c in code_lists]

    # -------------------------
    # Saving
    # -------------------------
    def save_audio(self, samples, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for i, audio_tensor in enumerate(samples):
            audio_np = audio_tensor.squeeze().cpu().detach().numpy()
            path = os.path.join(output_dir, f"output_{i}.wav")

            sf.write(path, audio_np, self.sample_rate)
            print(f"Saved: {path}")

    # -------------------------
    # Bytes synthesis (single text → WAV bytes, no disk I/O)
    # -------------------------
    def synthesize_bytes(self, text: str, voice: str = "tara") -> bytes:
        modified = self.build_inputs([text], voice)
        padded, mask = self.pad_inputs(modified)
        generated = self.generate(padded, mask)
        cropped = self.crop_after_token(generated)
        processed = self.process_rows(cropped)
        samples = self.decode_all(processed)
        if not samples:
            return b""
        audio_np = samples[0].squeeze().cpu().detach().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, self.sample_rate, format="WAV")
        return buf.getvalue()

    # -------------------------
    # Full pipeline
    # -------------------------
    def run(self, prompts, voice="tara", output_dir="./output"):
        modified = self.build_inputs(prompts, voice)
        padded, mask = self.pad_inputs(modified)

        generated = self.generate(padded, mask)
        cropped = self.crop_after_token(generated)
        processed = self.process_rows(cropped)

        samples = self.decode_all(processed)
        self.save_audio(samples, output_dir)

        return samples