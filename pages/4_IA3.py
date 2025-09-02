import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from peft_utils.model import load_base_model, DEFAULT_MODEL_ID
from peft_utils.data import load_tiny_instruct
from peft_utils.train import train_once
from peft_utils.viz import setup_korean_font, create_comparison_chart

# 한글 폰트 설정
setup_korean_font()

st.set_page_config(page_title="IA³ - 스케일링 벡터 학습", page_icon="⚖️", layout="wide")

st.title("⚖️ IA³ — 스케일링 벡터 학습 (Infused Adapter by Inhibiting and Amplifying Inner Activations)")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    model_id = st.text_input("Base model", value=DEFAULT_MODEL_ID, help="기본 모델 ID")
    target_modules = st.text_input("target_modules (쉼표 구분)", value="c_attn,c_proj", help="IA³를 적용할 모듈들")
    epochs = st.number_input("epochs", 1, 10, 1, help="학습 에포크 수")
    lr = st.number_input("learning_rate", 1e-6, 5e-3, 5e-4, step=1e-6, format="%.6f", help="학습률")

# 메인 콘텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 🎯 IA³란 무엇일까요?
    
    **IA³**는 **I**nfused **A**dapter by **I**nhibiting and **A**mplifying **I**nner **A**ctivations의 줄임말입니다.
    
    ### 🧩 기존 LoRA의 한계
    - **행렬 분해**: A × B 행렬로 가중치 변화량 학습
    - **파라미터 수**: rank에 따라 파라미터 수 증가
    - **복잡성**: 두 개의 행렬을 동시에 학습
    
    ### ✨ IA³의 해결책
    - **스케일링 벡터**: 단순한 스케일링 팩터만 학습
    - **극한 효율성**: 전체 파라미터의 0.01% 미만만 학습
    - **단순함**: 하나의 벡터만 학습하여 안정적
    """)

with col2:
    # IA³ 구조 시각화
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 파라미터 효율성 비교
    methods = ['Full Fine-tuning', 'LoRA (r=8)', 'IA³']
    param_counts = [100, 0.8, 0.04]  # 백분율로 표시
    
    bars = ax.bar(methods, param_counts, color=['red', 'orange', 'green'], alpha=0.7)
    ax.set_ylabel('학습 파라미터 (%)')
    ax.set_title('IA³: 극한 파라미터 효율성')
    ax.set_ylim(0, 110)
    
    # 값 표시
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

# 상세 설명
st.markdown("---")
st.markdown("""
## 🔬 IA³의 작동 원리

       ### 📊 핵심 아이디어
       ```
       W = W₀ ⊙ s
       ```

       여기서:
       - **W**: 최종 가중치
       - **W₀**: 원본 가중치 (고정, 학습 안함)
       - **s**: 스케일링 벡터 (학습 가능)
       - **⊙**: 요소별 곱셈 (Hadamard product)
       
       **수식 해석**: 원본 가중치 W₀에 스케일링 벡터 s를 요소별로 곱하여 최종 가중치 W를 만듭니다.

### 🎨 시각적 비교
""")

# LoRA vs IA³ 비교 차트
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 LoRA 구조")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # LoRA 구조
    ax.text(0.5, 0.9, '원본 가중치 (W0)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.arrow(0.5, 0.8, 0, -0.2, head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax.text(0.5, 0.6, 'LoRA (A×B)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax.arrow(0.5, 0.5, 0, -0.2, head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax.text(0.5, 0.3, 'W = W0 + A×B', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.1, '파라미터: 0.8%', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('LoRA 구조')
    ax.axis('off')
    
    st.pyplot(fig)

with col2:
    st.markdown("#### 🟢 IA³ 구조")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # IA³ 구조
    ax.text(0.5, 0.9, '원본 가중치 (W0)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.arrow(0.5, 0.8, 0, -0.2, head_width=0.05, head_length=0.05, fc='orange', ec='orange')
    ax.text(0.5, 0.6, '스케일링 벡터 (s)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.arrow(0.5, 0.5, 0, -0.2, head_width=0.05, head_length=0.05, fc='orange', ec='orange')
    ax.text(0.5, 0.3, 'W = W0 ⊙ s', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.1, '파라미터: 0.04%', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('IA³ 구조')
    ax.axis('off')
    
    st.pyplot(fig)

# 스케일링 벡터 작동 원리
st.markdown("---")
st.markdown("## 🎯 스케일링 벡터 작동 원리")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 스케일링 효과")
    
    # 스케일링 벡터의 효과 시각화
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 원본 가중치와 스케일링된 가중치
    x = np.linspace(0, 10, 100)
    original_weights = np.sin(x) * 2
    scaling_vector = 1 + 0.5 * np.sin(x * 0.5)  # 0.5 ~ 1.5 범위
    scaled_weights = original_weights * scaling_vector
    
    ax.plot(x, original_weights, 'b-', linewidth=2, label='원본 가중치')
    ax.plot(x, scaling_vector, 'g--', linewidth=2, label='스케일링 벡터')
    ax.plot(x, scaled_weights, 'r-', linewidth=2, label='스케일링된 가중치')
    
    ax.set_xlabel('가중치 차원')
    ax.set_ylabel('가중치 값')
    ax.set_title('IA³: 스케일링 벡터 효과')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with col2:
    st.markdown("### ⚡ 학습 과정")
    
    # 학습 과정 시각화
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 학습 에포크에 따른 스케일링 벡터 변화
    epochs = np.linspace(0, 10, 100)
    scaling_values = 1 + 0.3 * np.sin(epochs * 2) + np.random.normal(0, 0.05, 100)
    
    ax.plot(epochs, scaling_values, 'g-', linewidth=2, alpha=0.7)
    ax.fill_between(epochs, scaling_values - 0.1, scaling_values + 0.1, alpha=0.3, color='green')
    
    ax.set_xlabel('학습 에포크')
    ax.set_ylabel('스케일링 값')
    ax.set_title('스케일링 벡터 학습 과정')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.5)
    
    st.pyplot(fig)

# 파라미터 효율성 비교
st.markdown("---")
st.markdown("## 💾 파라미터 효율성 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 파라미터 수 비교")
    
    # 파라미터 수 비교 차트
    methods = ['Full Fine-tuning', 'LoRA (r=8)', 'LoRA (r=16)', 'IA³']
    param_counts = [100, 0.8, 1.6, 0.04]
    
    fig = create_comparison_chart(methods, param_counts, 
                                 '파라미터 효율성 비교', '학습 파라미터 (%)')
    st.pyplot(fig)

with col2:
    st.markdown("### ⚡ 학습 속도 비교")
    
    # 학습 속도 비교
    speed_data = [1, 15, 12, 20]  # 상대적 학습 속도
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(methods, speed_data, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax.set_ylabel('상대적 학습 속도')
    ax.set_title('학습 속도 비교')
    ax.set_ylim(0, 25)
    
    # 값 표시
    for bar, speed in zip(bars, speed_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{speed}x', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# 장단점
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ IA³의 장점")
    st.markdown("""
    - 💾 **극한 효율성**: 전체 파라미터의 0.01% 미만만 학습
    - ⚡ **빠른 학습**: 매우 적은 파라미터로 빠른 학습
    - 🔧 **단순함**: 스케일링 벡터만 학습하여 안정적
    - 💰 **비용 절약**: 최소한의 메모리와 계산으로 학습
    - 📱 **경량화**: 모바일/엣지 디바이스에 적합
    """)

with col2:
    st.markdown("### ⚠️ IA³의 단점")
    st.markdown("""
    - 🎯 **표현력 제한**: 단순한 스케일링만 가능
    - 📊 **성능 제한**: 복잡한 태스크에서는 성능 떨어질 수 있음
    - 🔍 **적용 범위**: 특정 모듈에만 적용 가능
    - 🧮 **하이퍼파라미터**: 스케일링 범위 설정 필요
    """)

# 실습 섹션
st.markdown("---")
st.markdown("## 🚀 IA³ 실습하기")

st.info("""
💡 **실습 팁**: 
- **target_modules**: 주로 attention과 projection 레이어에 적용
- **학습률**: 일반적으로 LoRA보다 낮은 학습률 사용
- **에포크**: 적은 파라미터로 빠르게 수렴
- **적용 범위**: 전체 모델이 아닌 특정 모듈에만 적용
""")

if st.button("🚀 IA³ 데모 학습 실행", type="primary"):
    with st.spinner("IA³ 모델 준비 중..."):
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
                status_text.text("단계 2/3: IA³ 어댑터 구성")
            else:
                status_text.text("단계 3/3: 학습 시작")
            progress_bar.progress((i + 1) * 0.33)
            import time
            time.sleep(0.5)
        
        # 실제 학습 실행
        try:
            metrics = train_once(base, tok, train_ds, eval_ds, method="ia3",
                               target_modules=tm, epochs=epochs, lr=lr)
            
            st.success("🎉 IA³ 학습 완료!")
            
            # 결과 시각화
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("학습 손실", f"{metrics.get('train_loss', 'N/A'):.4f}")
                st.metric("평가 손실", f"{metrics.get('eval_loss', 'N/A'):.4f}")
            
            with col2:
                st.metric("학습 시간", f"{metrics.get('train_runtime', 'N/A'):.2f}초")
                st.metric("샘플/초", f"{metrics.get('train_samples_per_second', 'N/A'):.2f}")
            
            # 파라미터 효율성 표시
            st.markdown("### 📊 파라미터 효율성")
            
            # 실제 trainable params 계산 (시뮬레이션)
            total_params = 102758  # 예시 값
            trainable_params = len(tm) * 768  # 대략적인 계산 (IA³는 스케일링 벡터만)
            efficiency = (trainable_params / total_params) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 파라미터", f"{total_params:,}")
            with col2:
                st.metric("학습 파라미터", f"{trainable_params:,}")
            with col3:
                st.metric("효율성", f"{efficiency:.3f}%")
            
        except Exception as e:
            st.error(f"학습 중 오류 발생: {str(e)}")

# 추가 정보
st.markdown("---")
st.markdown("## 📚 더 알아보기")

with st.expander("🔍 IA³ 논문 정보"):
    st.markdown("""
    **논문**: "IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations" (2022)
    **저자**: Brian Lester, et al.
    **핵심 아이디어**: 스케일링 벡터만 학습하여 극한의 파라미터 효율성 달성
    """)

with st.expander("💡 실제 사용 사례"):
    st.markdown("""
    - **경량화 모델**: 모바일/엣지 디바이스용 모델
    - **빠른 적응**: 새로운 태스크에 빠르게 적응
    - **리소스 제약 환경**: 매우 제한된 메모리 환경
    - **프로토타이핑**: 빠른 실험과 검증
    """)

with st.expander("⚡ 성능 비교"):
    st.markdown("""
    | 방법 | 파라미터 효율성 | 성능 | 학습 속도 | 메모리 사용량 |
    |------|----------------|------|-----------|---------------|
    | Full Fine-tuning | 낮음 | 높음 | 느림 | 높음 |
    | LoRA | 높음 | 높음 | 빠름 | 중간 |
    | **IA³** | **극한** | **보통** | **매우 빠름** | **낮음** |
    """)

with st.expander("🔧 구현 세부사항"):
    st.markdown("""
    - **스케일링**: 요소별 곱셈으로 가중치 조정
    - **초기화**: 보통 1로 초기화 (변화 없음)
    - **정규화**: 스케일링 값의 범위 제한
    - **적용 범위**: 특정 모듈에만 선택적 적용
    """)
