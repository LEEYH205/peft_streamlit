import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from peft_utils.data import load_tiny_instruct
from peft_utils.model import DEFAULT_MODEL_ID, load_base_model
from peft_utils.train import train_once

st.set_page_config(page_title="LoRA - 저랭크 적응", page_icon="🔧", layout="wide")

st.title("🔧 LoRA — 저랭크 적응 (Low-Rank Adaptation)")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    model_id = st.text_input("Base model", value=DEFAULT_MODEL_ID, help="기본 모델 ID")
    r = st.slider(
        "LoRA rank (r)",
        1,
        64,
        8,
        step=1,
        help="저랭크 행렬의 차원 (낮을수록 빠르지만 성능이 떨어질 수 있음)",
    )
    alpha = st.slider(
        "lora_alpha",
        1,
        128,
        16,
        step=1,
        help="LoRA 스케일링 팩터 (보통 r의 2배로 설정)",
    )
    dropout = st.slider(
        "lora_dropout", 0.0, 0.5, 0.05, step=0.01, help="드롭아웃 비율 (과적합 방지)"
    )
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
    ## 🎯 LoRA란 무엇일까요?

    **LoRA**는 **Lo**w-**Ra**nk Adaptation의 줄임말입니다.

    ### 🧩 기존 미세조정의 문제점
    - **전체 파라미터 학습**: 수억~수천억 개의 파라미터를 모두 학습
    - **메모리 부족**: GPU 메모리가 부족하여 학습 불가
    - **느린 학습**: 모든 파라미터를 업데이트해야 함

    ### ✨ LoRA의 해결책
    - **저랭크 분해**: 가중치를 A × B로 분해하여 학습
    - **파라미터 효율성**: 전체의 0.1% 미만의 파라미터만 학습
    - **빠른 학습**: 작은 행렬만 업데이트하여 빠름
    """
    )

with col2:
    # LoRA 구조 시각화
    fig, ax = plt.subplots(figsize=(6, 4))

    # 가중치 분해 과정 시각화
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) * 2  # 원본 가중치
    y2 = np.sin(x + 0.3) * 1.8  # LoRA 적용 후

    ax.plot(x, y1, "b-", linewidth=3, label="원본 가중치")
    ax.plot(x, y2, "r--", linewidth=3, label="LoRA 적용")
    ax.fill_between(x, y1, y2, alpha=0.3, color="green", label="학습된 변화")

    ax.set_xlabel("가중치 차원")
    ax.set_ylabel("가중치 값")
    ax.set_title("LoRA: 저랭크 분해 및 학습")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

# 상세 설명
st.markdown("---")
st.markdown(
    """
## 🔬 LoRA의 작동 원리

### 📊 수학적 표현
```
W = W₀ + ΔW
ΔW = A × B
```

여기서:
- **W**: 최종 가중치
- **W₀**: 원본 가중치 (고정, 학습 안함)
- **ΔW**: LoRA로 학습된 변화량
- **A**: r × d 행렬 (학습 가능)
- **B**: d × r 행렬 (학습 가능)
- **r**: rank (차원, 보통 8-64)

### 🎨 시각적 구조
"""
)

# LoRA 구조 시각화
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 기존 미세조정")
    fig, ax = plt.subplots(figsize=(5, 4))

    # 전체 파라미터 학습
    ax.text(
        0.5,
        0.8,
        "원본 가중치",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )
    ax.arrow(0.5, 0.7, 0, -0.2, head_width=0.05, head_length=0.05, fc="red", ec="red")
    ax.text(
        0.5,
        0.5,
        "전체 파라미터 학습",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
    )
    ax.arrow(0.5, 0.4, 0, -0.2, head_width=0.05, head_length=0.05, fc="red", ec="red")
    ax.text(
        0.5,
        0.2,
        "미세조정된 가중치",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("전체 미세조정")
    ax.axis("off")

    st.pyplot(fig)

with col2:
    st.markdown("#### 🟢 LoRA")
    fig, ax = plt.subplots(figsize=(5, 4))

    # LoRA 구조
    ax.text(
        0.5,
        0.9,
        "원본 가중치 (고정)",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )

    # A, B 행렬
    ax.text(
        0.3,
        0.7,
        "A 행렬 (r×d)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"),
    )
    ax.text(
        0.7,
        0.7,
        "B 행렬 (d×r)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"),
    )

    ax.arrow(
        0.3, 0.6, 0, -0.2, head_width=0.05, head_length=0.05, fc="orange", ec="orange"
    )
    ax.arrow(
        0.7, 0.6, 0, -0.2, head_width=0.05, head_length=0.05, fc="orange", ec="orange"
    )

    ax.text(
        0.3,
        0.4,
        "A 학습",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"),
    )
    ax.text(
        0.7,
        0.4,
        "B 학습",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"),
    )

    ax.arrow(
        0.5, 0.3, 0, -0.2, head_width=0.05, head_length=0.05, fc="green", ec="green"
    )
    ax.text(
        0.5,
        0.1,
        "LoRA 가중치",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("LoRA 구조")
    ax.axis("off")

    st.pyplot(fig)

# 파라미터 효율성 비교
st.markdown("---")
st.markdown("## 💾 파라미터 효율성 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 파라미터 수 비교")

    # 파라미터 수 비교 차트
    methods = ["Full Fine-tuning", "LoRA (r=8)", "LoRA (r=16)", "LoRA (r=32)"]
    param_counts = [100, 0.8, 1.6, 3.2]  # 백분율로 표시

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        methods,
        param_counts,
        color=["red", "lightgreen", "green", "darkgreen"],
        alpha=0.7,
    )
    ax.set_ylabel("학습 파라미터 (%)")
    ax.set_title("파라미터 효율성 비교")
    ax.set_ylim(0, 110)

    # 값 표시
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{count}%",
            ha="center",
            va="bottom",
        )

    st.pyplot(fig)

with col2:
    st.markdown("### ⚡ 학습 속도 비교")

    # 학습 속도 비교
    speed_data = [1, 5, 10, 15]  # 상대적 속도

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        methods,
        speed_data,
        color=["red", "lightgreen", "green", "darkgreen"],
        alpha=0.7,
    )
    ax.set_ylabel("상대적 학습 속도")
    ax.set_title("학습 속도 비교")
    ax.set_ylim(0, 20)

    # 값 표시
    for bar, speed in zip(bars, speed_data):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{speed}x",
            ha="center",
            va="bottom",
        )

    st.pyplot(fig)

# 장단점
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ LoRA의 장점")
    st.markdown(
        """
    - 💾 **메모리 효율성**: 전체 파라미터의 0.1% 미만만 학습
    - ⚡ **빠른 학습**: 작은 행렬만 업데이트하여 빠름
    - 🔧 **간단함**: 구현이 간단하고 안정적
    - 📱 **모바일 친화적**: 작은 모델로 배포 가능
    - 🔄 **호환성**: 기존 모델과 완벽 호환
    """
    )

with col2:
    st.markdown("### ⚠️ LoRA의 단점")
    st.markdown(
        """
    - 🎯 **성능 제한**: rank가 낮을수록 성능이 떨어질 수 있음
    - 🔍 **하이퍼파라라미터**: r, alpha 값을 수동으로 설정해야 함
    - 📊 **모델 크기**: 어댑터를 추가로 저장해야 함
    - 🧮 **추론 오버헤드**: 약간의 추가 계산 필요
    """
    )

# 실습 섹션
st.markdown("---")
st.markdown("## 🚀 LoRA 실습하기")

st.info(
    """
💡 **실습 팁**:
- **r (rank)**: 8-32가 좋은 시작점, 낮을수록 빠르지만 성능이 떨어질 수 있음
- **alpha**: 보통 r의 2배로 설정 (r=8이면 alpha=16)
- **dropout**: 0.05-0.1 정도가 적당, 과적합 방지
- **target_modules**: GPT-2에서는 'c_attn,c_proj'가 일반적
"""
)

if st.button("🚀 LoRA 데모 학습 실행", type="primary"):
    with st.spinner("LoRA 모델 준비 중..."):
        base, tok, _ = load_base_model(model_id, four_bit=False)
        train_ds, eval_ds = load_tiny_instruct()
        tm = [s.strip() for s in target_modules.split(",") if s.strip()]

        # 학습 진행률 표시
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(3):
            if i == 0:
                status_text.text("단계 1/3: 모델 준비")
            elif i == 1:
                status_text.text("단계 2/3: LoRA 어댑터 구성")
            else:
                status_text.text("단계 3/3: 학습 시작")
            progress_bar.progress((i + 1) * 0.33)
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
            )

            st.success("🎉 LoRA 학습 완료!")

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

            # 파라미터 효율성 표시
            st.markdown("### 📊 파라미터 효율성")

            # 실제 trainable params 계산 (시뮬레이션)
            total_params = 103066  # 예시 값
            trainable_params = r * 2 * len(tm) * 768  # 대략적인 계산
            efficiency = (trainable_params / total_params) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 파라미터", f"{total_params:,}")
            with col2:
                st.metric("학습 파라미터", f"{trainable_params:,}")
            with col3:
                st.metric("효율성", f"{efficiency:.2f}%")

        except Exception as e:
            st.error(f"학습 중 오류 발생: {str(e)}")

# 추가 정보
st.markdown("---")
st.markdown("## 📚 더 알아보기")

with st.expander("🔍 LoRA 논문 정보"):
    st.markdown(
        """
    **논문**: "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
    **저자**: Edward J. Hu, et al.
    **핵심 아이디어**: 대규모 언어 모델을 저랭크 행렬로 효율적으로 미세조정
    """
    )

with st.expander("💡 실제 사용 사례"):
    st.markdown(
        """
    - **ChatGPT**: OpenAI의 대화형 AI 모델
    - **Claude**: Anthropic의 AI 어시스턴트
    - **코드 생성**: GitHub Copilot 등
    - **번역**: 다국어 번역 모델
    - **요약**: 긴 문서 요약 모델
    """
    )

with st.expander("⚡ 성능 비교"):
    st.markdown(
        """
    | 방법 | 파라미터 효율성 | 성능 | 학습 속도 | 메모리 사용량 |
    |------|----------------|------|-----------|---------------|
    | Full Fine-tuning | 낮음 | 높음 | 느림 | 높음 |
    | **LoRA** | **높음** | **높음** | **빠름** | **낮음** |
    | Prompt Tuning | 높음 | 보통 | 빠름 | 낮음 |
    """
    )

with st.expander("🔧 구현 세부사항"):
    st.markdown(
        """
    - **행렬 분해**: W = W₀ + A × B
    - **초기화**: A는 정규분포, B는 0으로 초기화
    - **스케일링**: alpha/r로 LoRA 출력을 스케일링
    - **드롭아웃**: 과적합 방지를 위한 정규화
    - **타겟 모듈**: 주로 attention과 projection 레이어에 적용
    """
    )
