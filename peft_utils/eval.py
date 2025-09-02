import math, torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def perplexity(model, tok, input_data, batch_size=2, max_len=256, device=None):
    """
    모델의 perplexity를 계산합니다.
    
    Args:
        model: 평가할 모델
        tok: 토크나이저
        input_data: 입력 데이터 (string 또는 dataset)
        batch_size: 배치 크기
        max_len: 최대 시퀀스 길이
        device: 디바이스
    
    Returns:
        float: perplexity 값
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # 입력 데이터 타입 확인 및 처리
    if isinstance(input_data, str):
        # 단일 텍스트인 경우
        return _perplexity_single_text(model, tok, input_data, max_len, device)
    else:
        # dataset인 경우
        return _perplexity_dataset(model, tok, input_data, batch_size, max_len, device)

def _perplexity_single_text(model, tok, text, max_len=256, device=None):
    """단일 텍스트의 perplexity 계산"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # 텍스트 토크나이제이션
    inputs = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    
    # 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs.loss.detach().float()
    
    return math.exp(loss.item()) if loss.item() < 20 else float("inf")

def _perplexity_dataset(model, tok, ds, batch_size=2, max_len=256, device=None):
    """데이터셋의 perplexity 계산"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    def tok_fx(ex):
        out = tok(ex["text"], truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        out["labels"] = out["input_ids"].clone()
        return out
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, 
                       collate_fn=lambda batch: tok_fx({"text":[b["text"] for b in batch]}))
    
    total_loss, total_tokens = 0.0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss.detach().float()
            total_loss += loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()
    
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss) if avg_loss < 20 else float("inf")
