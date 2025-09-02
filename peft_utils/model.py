import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"

def load_base_model(model_id=DEFAULT_MODEL_ID, four_bit=False, device_map="auto"):
    quant_ok = False
    bnb_config = None
    if four_bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                quant_ok = True
        except Exception:
            quant_ok = False

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if quant_ok and bnb_config is not None:
        base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device_map)
    else:
        base = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return base, tok, quant_ok
