import math
import os

import streamlit as st
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# PEFT 설정 지원 여부 확인
try:
    from peft import (
        IA3Config,
        LoraConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    PEFT_SUPPORTED = True
except ImportError as e:
    st.error(f"PEFT 라이브러리 로드 실패: {e}")
    PEFT_SUPPORTED = False


def _tokenize(ds, tok, max_len=256):
    def fx(ex):
        out = tok(ex["text"], truncation=True, padding="max_length", max_length=max_len)
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(fx, batched=True, remove_columns=["text"])


def _trainer(
    base,
    tok,
    train_ds,
    eval_ds,
    out_dir,
    lr=5e-4,
    epochs=1,
    bs=2,
    grad_accum=4,
    fp16=True,
):
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=50,
        save_total_limit=1,
        fp16=fp16 and torch.cuda.is_available(),
        bf16=not fp16 and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    return Trainer(
        model=base,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )


def build_adapter(
    base,
    method="lora",
    target_modules=None,
    r=8,
    alpha=16,
    dropout=0.05,
    num_virtual_tokens=16,
):
    if not PEFT_SUPPORTED:
        st.error("PEFT 라이브러리가 지원되지 않습니다.")
        return base

    if target_modules is None:
        # GPT2 family common names
        target_modules = ["c_attn", "c_proj"]

    try:
        if method == "lora":
            cfg = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
        elif method == "ia3":
            cfg = IA3Config(task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
        elif method == "prefix":
            # Prefix Tuning에서 캐시 문제를 피하기 위해 추가 설정
            try:
                cfg = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=num_virtual_tokens,
                    prefix_projection=False,  # 캐시 문제 방지
                    encoder_hidden_size=(
                        base.config.hidden_size
                        if hasattr(base.config, "hidden_size")
                        else 768
                    ),
                )
            except Exception as e:
                st.warning(f"Prefix Tuning 설정 실패: {e}. LoRA로 대체합니다.")
                cfg = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    target_modules=target_modules,
                    task_type=TaskType.CAUSAL_LM,
                )
        elif method == "prompt":
            try:
                cfg = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens
                )
            except Exception as e:
                st.warning(f"Prompt Tuning 설정 실패: {e}. LoRA로 대체합니다.")
                cfg = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    target_modules=target_modules,
                    task_type=TaskType.CAUSAL_LM,
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        return get_peft_model(base, cfg)
    except Exception as e:
        st.error(f"어댑터 생성 실패: {e}. 기본 모델을 반환합니다.")
        return base


def train_once(base, tok, train_ds, eval_ds, out_dir="outputs/demo", **kwargs):
    os.makedirs(out_dir, exist_ok=True)
    method = kwargs.pop("method", "lora")
    st.write(f"### Adapter: {method}")

    if kwargs.get("four_bit", False):
        st.info("4-bit 준비 중… k-bit 학습 준비를 수행합니다.")
        try:
            base = prepare_model_for_kbit_training(base)
        except Exception as e:
            st.warning(f"4-bit 양자화 실패: {e}. 일반 모델로 진행합니다.")

    # 데이터 토크나이제이션
    st.write("데이터 토크나이제이션 중...")
    try:
        tokenized_train_ds = _tokenize(train_ds, tok)
        tokenized_eval_ds = _tokenize(eval_ds, tok)
    except Exception as e:
        st.error(f"데이터 토크나이제이션 실패: {e}")
        return {"error": str(e)}

    # build_adapter에 필요한 인자만 추출
    adapter_kwargs = {}
    if method == "lora":
        adapter_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["r", "alpha", "dropout", "target_modules"]
        }
    elif method == "ia3":
        adapter_kwargs = {k: v for k, v in kwargs.items() if k in ["target_modules"]}
    elif method in ["prefix", "prompt"]:
        adapter_kwargs = {
            k: v for k, v in kwargs.items() if k in ["num_virtual_tokens"]
        }

    try:
        model = build_adapter(base, method=method, **adapter_kwargs)
        st.write(model.print_trainable_parameters())
    except Exception as e:
        st.error(f"어댑터 생성 실패: {e}")
        return {"error": str(e)}

    try:
        tr = _trainer(
            model,
            tok,
            tokenized_train_ds,
            tokenized_eval_ds,
            out_dir,
            lr=kwargs.get("lr", 5e-4),
            epochs=kwargs.get("epochs", 1),
        )
        tr.train()
        metrics = tr.evaluate()

        # metrics에서 eval_loss 안전하게 추출
        eval_loss = None
        if isinstance(metrics, dict) and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            # 배열인 경우 첫 번째 요소 사용
            if hasattr(eval_loss, "__len__") and len(eval_loss) > 0:
                eval_loss = (
                    eval_loss[0]
                    if hasattr(eval_loss, "__getitem__")
                    else float(eval_loss)
                )
            else:
                eval_loss = float(eval_loss)

        ppl = math.exp(eval_loss) if eval_loss is not None else None
        if ppl is not None:
            st.success(f"평가 완료. metrics={metrics}, PPL={ppl:.2f}")
        else:
            st.success(f"평가 완료. metrics={metrics}")

        # 어댑터 저장
        try:
            model.save_pretrained(os.path.join(out_dir, f"{method}_adapter"))
        except Exception as e:
            st.warning(f"어댑터 저장 실패: {e}")

        return metrics
    except Exception as e:
        st.error(f"학습 실패: {e}")
        return {"error": str(e)}
