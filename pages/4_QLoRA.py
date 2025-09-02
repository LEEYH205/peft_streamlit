import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from peft_utils.data import load_tiny_instruct
from peft_utils.model import DEFAULT_MODEL_ID, load_base_model
from peft_utils.train import train_once
from peft_utils.viz import create_comparison_chart, setup_korean_font

# 한글 폰트 설정
setup_korean_font()

st.set_page_config(
    page_title="QLoRA - 4-bit 양자화 + LoRA", page_icon="🧱", layout="wide"
)

st.title("🧱 QLoRA — 4-bit 양자화 + LoRA (Quantized LoRA)")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    model_id = st.text_input("Base model", value=DEFAULT_MODEL_ID, help="기본 모델 ID")
    r = st.slider("LoRA rank (r)", 1, 64, 8, step=1, help="저랭크 행렬의 차원")
    alpha = st.slider("lora_alpha", 1, 128, 16, step=1, help="LoRA 스케일링 팩터")
    dropout = st.slider("lora_dropout", 0.0, 0.5, 0.05, step=0.01, help="드롭아웃 비율")
    target_modules = st.text_input(
        "target_modules (쉼표 구분)", value="c_attn,c_proj", help="LoRA를 적용할 모듈들"
    )
    epochs = st.number_input("epochs", 1, 10, 1, help="학습 에포크 수")
    lr = st.number_input(
        "learning_rate", 1e-6, 5e-3, 5e-4, step=1e-6, format="%.6f", help="학습률"
    )

# 메인 콘텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
    ## 🎯 QLoRA란 무엇일까요?

    **QLoRA**는 **Q**uantized + **LoRA**의 줄임말입니다.

    ### 🧩 기존 LoRA의 한계
    - **메모리 부족**: 큰 모델을 로드할 때 GPU 메모리 부족
    - **하드웨어 요구**: 고사양 GPU가 필요
    - **비용**: 비싼 GPU 사용 필요

    ### ✨ QLoRA의 해결책
    - **4-bit 양자화**: 가중치를 4비트로 압축하여 메모리 절약
    - **메모리 효율성**: 기존 대비 8배 메모리 절약
    - **성능 유지**: 양자화해도 성능 저하 최소화
    """
    )

with col2:
    # QLoRA 구조 시각화
    fig, ax = plt.subplots(figsize=(6, 4))

    # 메모리 사용량 비교
    methods = ["Full Model", "LoRA", "QLoRA"]
    memory_usage = [100, 20, 12.5]  # 상대적 메모리 사용량

    bars = ax.bar(methods, memory_usage, color=["red", "orange", "green"], alpha=0.7)
    ax.set_ylabel("상대적 메모리 사용량 (%)")
    ax.set_title("QLoRA: 메모리 효율성")

    # 값 표시
    for bar, usage in zip(bars, memory_usage):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{usage}%",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

# 상세 설명
st.markdown("---")
st.markdown(
    """
## 🔬 QLoRA의 작동 원리

### 📊 양자화 과정
```
1. 원본 가중치 (32-bit float)
2. 4-bit 양자화 (정밀도 손실)
3. LoRA 어댑터 추가 (학습 가능)
4. 추론 시 양자화 해제 + LoRA 적용
```

### 🎨 시각적 비교
"""
)

# 양자화 과정 시각화
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 일반 LoRA")
    fig, ax = plt.subplots(figsize=(5, 4))

    # 일반 LoRA 구조
    ax.text(
        0.5,
        0.9,
        "원본 가중치 (32-bit)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )
    ax.arrow(0.5, 0.8, 0, -0.2, head_width=0.05, head_length=0.05, fc="red", ec="red")
    ax.text(
        0.5,
        0.6,
        "LoRA 어댑터",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
    )
    ax.arrow(0.5, 0.5, 0, -0.2, head_width=0.05, head_length=0.05, fc="red", ec="red")
    ax.text(
        0.5,
        0.3,
        "메모리: 100%",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("일반 LoRA")
    ax.axis("off")

    st.pyplot(fig)

with col2:
    st.markdown("#### 🟢 QLoRA")
    fig, ax = plt.subplots(figsize=(5, 4))

    # QLoRA 구조
    ax.text(
        0.5,
        0.9,
        "원본 가중치 (32-bit)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )
    ax.arrow(
        0.5, 0.8, 0, -0.2, head_width=0.05, head_length=0.05, fc="orange", ec="orange"
    )
    ax.text(
        0.5,
        0.6,
        "4-bit 양자화",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )
    ax.arrow(
        0.5, 0.5, 0, -0.2, head_width=0.05, head_length=0.05, fc="orange", ec="orange"
    )
    ax.text(
        0.5,
        0.3,
        "LoRA 어댑터",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
    )
    ax.arrow(
        0.5, 0.2, 0, -0.2, head_width=0.05, head_length=0.05, fc="green", ec="green"
    )
    ax.text(
        0.5,
        0.0,
        "메모리: 12.5%",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("QLoRA")
    ax.axis("off")

    st.pyplot(fig)

# 양자화 수준별 비교
st.markdown("---")
st.markdown("## 💾 양자화 수준별 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 메모리 사용량 비교")

    # 양자화 수준별 메모리 사용량
    quantization_levels = [
        "32-bit (FP32)",
        "16-bit (FP16)",
        "8-bit (INT8)",
        "4-bit (INT4)",
    ]
    memory_usage = [100, 50, 25, 12.5]

    fig = create_comparison_chart(
        quantization_levels,
        memory_usage,
        "양자화 수준별 메모리 사용량",
        "메모리 사용량 (%)",
    )
    st.pyplot(fig)

with col2:
    st.markdown("### ⚡ 성능 vs 메모리 트레이드오프")

    # 성능 vs 메모리 트레이드오프
    fig, ax = plt.subplots(figsize=(6, 4))

    x = memory_usage
    y = [100, 98, 95, 90]  # 상대적 성능

    ax.plot(x, y, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("메모리 사용량 (%)")
    ax.set_ylabel("상대적 성능 (%)")
    ax.set_title("성능 vs 메모리 트레이드오프")
    ax.grid(True, alpha=0.3)

    # 점에 라벨 추가
    for i, (mem, perf) in enumerate(zip(x, y)):
        ax.annotate(
            f"{quantization_levels[i].split()[0]}",
            (mem, perf),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()
    st.pyplot(fig)

# 장단점
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ QLoRA의 장점")
    st.markdown(
        """
    - 💾 **메모리 절약**: 8배 메모리 절약
    - 🚀 **빠른 학습**: 큰 모델도 학습 가능
    - 💰 **비용 절약**: 저사양 GPU로도 학습
    - 📱 **접근성**: 더 많은 사람이 사용 가능
    - 🔧 **호환성**: 기존 LoRA와 동일한 사용법
    """
    )

with col2:
    st.markdown("### ⚠️ QLoRA의 단점")
    st.markdown(
        """
    - 🎯 **정밀도 손실**: 4-bit 양자화로 인한 정보 손실
    - 🧮 **추론 오버헤드**: 양자화 해제 과정 필요
    - 🔍 **하이퍼파라미터**: 양자화 설정 추가 필요
    - 📊 **안정성**: 극단적 양자화로 인한 불안정성
    """
    )

# 실습 섹션
st.markdown("---")
st.markdown("## 🚀 QLoRA 실습하기")

st.info(
    """
💡 **실습 팁**:
- **4-bit 양자화**: 메모리 절약이 최우선일 때 사용
- **r (rank)**: 8-32가 좋은 시작점
- **alpha**: 보통 r의 2배로 설정
- **주의**: Mac 환경에서는 4-bit 양자화가 지원되지 않을 수 있음
"""
)

if st.button("🚀 QLoRA 데모 학습 실행", type="primary"):
    with st.spinner("QLoRA 모델 준비 중..."):
        base, tok, quant_ok = load_base_model(model_id, four_bit=True)

        if not quant_ok:
            st.warning(
                "⚠️ 이 환경에서는 4-bit 양자화를 사용할 수 없어 일반 LoRA로 폴백합니다."
            )
            st.info(
                "💡 Mac 환경에서는 4-bit 양자화가 지원되지 않습니다. 대신 일반 LoRA를 사용합니다."
            )
        else:
            st.success("✅ 4-bit 양자화가 성공적으로 적용되었습니다!")

        train_ds, eval_ds = load_tiny_instruct()
        tm = [s.strip() for s in target_modules.split(",") if s.strip()]

        # 학습 진행률 표시
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(4):
            if i == 0:
                status_text.text("단계 1/4: 모델 준비")
            elif i == 1:
                status_text.text("단계 2/4: 양자화 적용")
            elif i == 2:
                status_text.text("단계 3/4: LoRA 어댑터 구성")
            else:
                status_text.text("단계 4/4: 학습 시작")
            progress_bar.progress((i + 1) * 0.25)
            import time

            time.sleep(0.5)

        # 실제 학습 실행
        try:
            metrics = train_once(
                base,
                tok,
                train_ds,
                eval_ds,
                method="lora",
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=tm,
                epochs=epochs,
                lr=lr,
                four_bit=quant_ok,
            )

            st.success("🎉 QLoRA 학습 완료!")

            # 결과 시각화
            col1, col2 = st.columns(2)

            with col1:
                st.metric("학습 손실", f"{metrics.get('train_loss', 'N/A'):.4f}")
                st.metric("평가 손실", f"{metrics.get('eval_loss', 'N/A'):.4f}")

            with col2:
                st.metric("학습 시간", f"{metrics.get('train_runtime', 'N/A'):.2f}초")
                st.metric(
                    "샘플/초", f"{metrics.get('train_samples_per_second', 'N/A'):.2f}"
                )

            # 메모리 효율성 표시
            st.markdown("### 💾 메모리 효율성")

            if quant_ok:
                efficiency = "4-bit 양자화 적용됨 (메모리 87.5% 절약)"
                color = "green"
            else:
                efficiency = "일반 LoRA (4-bit 양자화 미지원)"
                color = "orange"

            st.info(f"**메모리 효율성**: {efficiency}")

        except Exception as e:
            st.error(f"학습 중 오류 발생: {str(e)}")

# Mac 환경에서의 4-bit 양자화 제한사항
st.markdown("---")
st.markdown("## 🍎 Mac 환경에서의 4-bit 양자화 제한사항")

st.warning(
    """
⚠️ **중요**: Mac 환경에서는 4-bit 양자화가 지원되지 않습니다.
이는 하드웨어와 소프트웨어의 제약사항 때문입니다.
"""
)

with st.expander("🔍 Mac에서 4-bit 양자화가 안 되는 이유"):
    st.markdown(
        """
    ### 🖥️ 하드웨어 제약
    - **Apple Silicon (M1/M2/M3)**: CUDA 지원 안함
    - **Intel Mac**: GPU 메모리 제한
    - **Metal Performance Shaders (MPS)**: 양자화 라이브러리 미지원

    ### 📦 소프트웨어 제약
    - **bitsandbytes**: CUDA 전용 라이브러리
    - **GPTQ/AWQ**: NVIDIA GPU 최적화
    - **PEFT 양자화**: CUDA 기반 구현

    ### 🔧 대안 방법
    1. **일반 LoRA**: 메모리 효율성은 떨어지지만 안정적
    2. **IA³**: 극한의 파라미터 효율성
    3. **Prompt Tuning**: 최소한의 메모리 사용
    4. **클라우드 서비스**: GPU 서버 활용
    """
    )

with st.expander("💻 환경별 양자화 지원 현황"):
    st.markdown(
        """
    | 환경 | 4-bit 양자화 | 8-bit 양자화 | 16-bit 양자화 | 메모리 효율성 |
    |------|--------------|--------------|---------------|---------------|
    | **NVIDIA GPU** | ✅ 지원 | ✅ 지원 | ✅ 지원 | 매우 높음 |
    | **Apple Silicon** | ❌ 미지원 | ❌ 미지원 | ✅ 지원 | 중간 |
    | **Intel CPU** | ❌ 미지원 | ❌ 미지원 | ✅ 지원 | 낮음 |
    | **AMD GPU** | ⚠️ 제한적 | ⚠️ 제한적 | ✅ 지원 | 중간 |

    **결론**: Mac 환경에서는 일반 LoRA나 IA³를 사용하는 것이 최선입니다.
    """
    )

with st.expander("🚀 Mac에서 PEFT 최적화 팁"):
    st.markdown(
        """
    ### 📱 Apple Silicon 최적화
    - **MPS 가속**: Metal Performance Shaders 활용
    - **메모리 관리**: 효율적인 배치 크기 설정
    - **모델 크기**: 적절한 모델 선택

    ### 🔧 하이퍼파라미터 튜닝
    - **LoRA rank**: 8-16 범위에서 실험
    - **배치 크기**: 메모리에 맞게 조정
    - **학습률**: 안정적인 범위에서 설정

    ### 💾 메모리 절약 전략
    - **Gradient Checkpointing**: 메모리 절약
    - **Mixed Precision**: 16-bit 학습 활용
    - **모듈별 적용**: 필요한 레이어만 선택
    """
    )

# 추가 정보
st.markdown("---")
st.markdown("## 📚 더 알아보기")

with st.expander("🔍 QLoRA 논문 정보"):
    st.markdown(
        """
    **논문**: "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
    **저자**: Tim Dettmers, et al.
    **핵심 아이디어**: 4-bit 양자화와 LoRA를 결합하여 메모리 효율적인 미세조정
    """
    )

with st.expander("💡 실제 사용 사례"):
    st.markdown(
        """
    - **대규모 모델**: 7B, 13B, 70B 모델 학습
    - **리소스 제약 환경**: 개인 개발자, 연구자
    - **비용 효율적 학습**: 클라우드 비용 절약
    - **실험적 연구**: 다양한 모델 아키텍처 테스트
    """
    )

with st.expander("⚡ 성능 비교"):
    st.markdown(
        """
    | 방법 | 메모리 사용량 | 성능 | 학습 속도 | 하드웨어 요구 |
    |------|---------------|------|-----------|---------------|
    | Full Fine-tuning | 100% | 높음 | 느림 | 고사양 GPU |
    | LoRA | 20% | 높음 | 빠름 | 중간 GPU |
    | **QLoRA** | **12.5%** | **높음** | **빠름** | **저사양 GPU** |
    """
    )

with st.expander("🔧 구현 세부사항"):
    st.markdown(
        """
    - **양자화**: GPTQ, AWQ 등의 4-bit 양자화 기법
    - **LoRA**: 저랭크 어댑터로 학습 가능한 파라미터 추가
    - **메모리 관리**: 그래디언트 체크포인팅으로 메모리 절약
    - **안정화**: 양자화 노이즈를 LoRA로 보상
    """
    )
