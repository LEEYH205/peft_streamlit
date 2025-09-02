import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from peft_utils.model import load_base_model, DEFAULT_MODEL_ID
from peft_utils.data import load_tiny_instruct
from peft_utils.train import train_once
from peft_utils.viz import setup_korean_font, create_comparison_chart

# 한글 폰트 설정
setup_korean_font()

st.set_page_config(page_title="Prefix & Prompt Tuning - 가상 토큰 학습", page_icon="📌", layout="wide")

st.title("📌 Prefix & Prompt Tuning — 가상 토큰 학습")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    model_id = st.text_input("Base model", value=DEFAULT_MODEL_ID, help="기본 모델 ID")
    num_virtual_tokens = st.slider("가상 토큰 수", 1, 100, 16, step=1, help="추가할 가상 토큰의 개수")
    epochs = st.number_input("epochs", 1, 10, 1, help="학습 에포크 수")
    lr = st.number_input("learning_rate", 1e-6, 5e-3, 5e-4, step=1e-6, format="%.6f", help="학습률")
    
    st.markdown("---")
    st.markdown("### 🔧 방법 선택")
    method = st.selectbox("PEFT 방법", ["prefix", "prompt"], 
                         help="Prefix Tuning: 입력에 가상 토큰 추가\nPrompt Tuning: 입력 시작에만 가상 토큰 추가")

# 메인 콘텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 🎯 Prefix & Prompt Tuning이란 무엇일까요?
    
    **Prefix & Prompt Tuning**은 입력에 **가상(soft) 토큰**을 추가하여 태스크 특성을 주입합니다.
    
    ### 🧩 기존 방식의 한계
    - **하드코딩**: 프롬프트를 수동으로 작성해야 함
    - **일관성 부족**: 프롬프트마다 다른 결과
    - **최적화 불가**: 프롬프트를 학습할 수 없음
    
    ### ✨ 가상 토큰의 해결책
    - **학습 가능**: 프롬프트를 자동으로 최적화
    - **일관성**: 동일한 태스크에 일관된 결과
    - **효율성**: 모델 파라미터는 건드리지 않음
    """)

with col2:
    # 가상 토큰 구조 시각화
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 토큰 시퀀스 시각화
    tokens = ['[PREFIX]', '[PREFIX]', '[PREFIX]', 'Hello', 'world', '!']
    colors = ['red'] * 3 + ['blue'] * 3
    
    y_pos = np.arange(len(tokens))
    bars = ax.barh(y_pos, [1] * len(tokens), color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.set_xlabel('토큰 위치')
    ax.set_title('가상 토큰 + 실제 토큰')
    
    # 가상 토큰 표시
    ax.text(0.5, 1.5, '가상 토큰 (학습 가능)', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    ax.text(0.5, 4.5, '실제 토큰 (고정)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7))
    
    plt.tight_layout()
    st.pyplot(fig)

# 상세 설명
st.markdown("---")
st.markdown("""
## 🔬 작동 원리

### 📊 핵심 아이디어
```
입력: [PREFIX] [PREFIX] ... [PREFIX] + 실제 텍스트
출력: 태스크에 최적화된 응답
```

### 🎨 Prefix vs Prompt Tuning
""")

# Prefix vs Prompt Tuning 비교
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 Prefix Tuning")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Prefix Tuning 구조
    ax.text(0.5, 0.9, '입력 텍스트', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.arrow(0.5, 0.8, 0, -0.2, head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax.text(0.5, 0.6, '가상 토큰 추가', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax.arrow(0.5, 0.5, 0, -0.2, head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax.text(0.5, 0.3, '전체 모델 학습', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.1, '성능: 높음', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Prefix Tuning')
    ax.axis('off')
    
    st.pyplot(fig)

with col2:
    st.markdown("#### 🟢 Prompt Tuning")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Prompt Tuning 구조
    ax.text(0.5, 0.9, '입력 텍스트', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.arrow(0.5, 0.8, 0, -0.2, head_width=0.05, head_length=0.05, fc='orange', ec='orange')
    ax.text(0.5, 0.6, '시작에만 추가', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.arrow(0.5, 0.5, 0, -0.2, head_width=0.05, head_length=0.05, fc='orange', ec='orange')
    ax.text(0.5, 0.3, '가상 토큰만 학습', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.1, '효율성: 높음', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Prompt Tuning')
    ax.axis('off')
    
    st.pyplot(fig)

# 가상 토큰 학습 과정
st.markdown("---")
st.markdown("## 🎯 가상 토큰 학습 과정")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 학습 과정")
    
    # 학습 과정 시각화
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 학습 단계별 가상 토큰 변화
    steps = np.linspace(0, 10, 100)
    token1 = 0.5 + 0.5 * np.sin(steps * 1.5) + np.random.normal(0, 0.1, 100)
    token2 = 0.3 + 0.7 * np.sin(steps * 2.0) + np.random.normal(0, 0.1, 100)
    token3 = 0.7 + 0.3 * np.sin(steps * 1.0) + np.random.normal(0, 0.1, 100)
    
    ax.plot(steps, token1, 'r-', linewidth=2, label='가상 토큰 1', alpha=0.7)
    ax.plot(steps, token2, 'g-', linewidth=2, label='가상 토큰 2', alpha=0.7)
    ax.plot(steps, token3, 'b-', linewidth=2, label='가상 토큰 3', alpha=0.7)
    
    ax.set_xlabel('학습 단계')
    ax.set_ylabel('토큰 임베딩 값')
    ax.set_title('가상 토큰 학습 과정')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with col2:
    st.markdown("### ⚡ 파라미터 효율성")
    
    # 파라미터 효율성 비교
    methods = ['Full Fine-tuning', 'LoRA', 'Prefix Tuning', 'Prompt Tuning']
    param_counts = [100, 0.8, 0.1, 0.05]  # 백분율로 표시
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(methods, param_counts, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax.set_ylabel('학습 파라미터 (%)')
    ax.set_title('파라미터 효율성 비교')
    ax.set_ylim(0, 110)
    
    # 값 표시
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# 장단점
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ 가상 토큰의 장점")
    st.markdown("""
    - 🎯 **태스크 특화**: 특정 작업에 최적화된 프롬프트
    - 💾 **효율성**: 모델 파라미터는 건드리지 않음
    - 🔧 **유연성**: 다양한 태스크에 쉽게 적용
    - 📱 **경량화**: 매우 작은 어댑터만 저장
    - 🔄 **재사용**: 학습된 가상 토큰을 다른 모델에 적용
    """)

with col2:
    st.markdown("### ⚠️ 가상 토큰의 단점")
    st.markdown("""
    - 🎯 **성능 제한**: 복잡한 태스크에서는 성능 떨어질 수 있음
    - 📏 **길이 제한**: 가상 토큰 수에 따른 성능 제한
    - 🔍 **해석 어려움**: 가상 토큰의 의미 해석이 어려움
    - 🧮 **하이퍼파라미터**: 가상 토큰 수 설정 필요
    """)

# 실습 섹션
st.markdown("---")
st.markdown("## 🚀 Prefix & Prompt Tuning 실습하기")

st.info("""
💡 **실습 팁**: 
- **가상 토큰 수**: 16-64가 좋은 시작점
- **Prefix Tuning**: 더 나은 성능, 더 많은 파라미터
- **Prompt Tuning**: 더 효율적, 더 적은 파라미터
- **주의**: 현재 환경에서는 일부 기능이 제한될 수 있음
""")

if st.button("🚀 가상 토큰 데모 학습 실행", type="primary"):
    with st.spinner(f"{method.upper()} 모델 준비 중..."):
        try:
            base, tok, _ = load_base_model(model_id, four_bit=False)
            train_ds, eval_ds = load_tiny_instruct()
            
            # 학습 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(4):
                if i == 0:
                    status_text.text("단계 1/4: 모델 준비")
                elif i == 1:
                    status_text.text("단계 2/4: 가상 토큰 초기화")
                elif i == 2:
                    status_text.text("단계 3/4: 어댑터 구성")
                else:
                    status_text.text("단계 4/4: 학습 시작")
                progress_bar.progress((i + 1) * 0.25)
                import time
                time.sleep(0.5)
            
            # 실제 학습 실행
            metrics = train_once(base, tok, train_ds, eval_ds, method=method,
                               num_virtual_tokens=num_virtual_tokens, epochs=epochs, lr=lr)
            
            st.success(f"🎉 {method.upper()} 학습 완료!")
            
            # 결과 시각화
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("학습 손실", f"{metrics.get('train_loss', 'N/A'):.4f}")
                st.metric("평가 손실", f"{metrics.get('eval_loss', 'N/A'):.4f}")
            
            with col2:
                st.metric("학습 시간", f"{metrics.get('train_runtime', 'N/A'):.2f}초")
                st.metric("샘플/초", f"{metrics.get('train_samples_per_second', 'N/A'):.2f}")
            
            # 가상 토큰 정보 표시
            st.markdown("### 📌 가상 토큰 정보")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("가상 토큰 수", num_virtual_tokens)
            with col2:
                st.metric("학습 파라미터", f"{num_virtual_tokens * 768:,}")
            with col3:
                st.metric("방법", method.upper())
            
        except Exception as e:
            st.error(f"학습 중 오류 발생: {str(e)}")
            st.info("💡 현재 환경에서 Prefix/Prompt Tuning이 지원되지 않을 수 있습니다. LoRA나 IA³를 대신 사용해보세요.")

# 추가 정보
st.markdown("---")
st.markdown("## 📚 더 알아보기")

with st.expander("🔍 논문 정보"):
    st.markdown("""
    **Prefix Tuning**: "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021)
    **Prompt Tuning**: "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021)
    **핵심 아이디어**: 입력에 학습 가능한 가상 토큰을 추가하여 태스크 특화
    """)

with st.expander("💡 실제 사용 사례"):
    st.markdown("""
    - **번역**: 언어별 최적화된 프롬프트
    - **요약**: 문서 요약 태스크 최적화
    - **대화**: 대화형 AI 성격 설정
    - **코드 생성**: 프로그래밍 언어별 최적화
    """)

with st.expander("⚡ 성능 비교"):
    st.markdown("""
    | 방법 | 파라미터 효율성 | 성능 | 학습 속도 | 메모리 사용량 |
    |------|----------------|------|-----------|---------------|
    | Full Fine-tuning | 낮음 | 높음 | 느림 | 높음 |
    | LoRA | 높음 | 높음 | 빠름 | 중간 |
    | **Prefix Tuning** | **높음** | **높음** | **빠름** | **낮음** |
    | **Prompt Tuning** | **매우 높음** | **보통** | **매우 빠름** | **매우 낮음** |
    """)

with st.expander("🔧 구현 세부사항"):
    st.markdown("""
    - **가상 토큰**: 학습 가능한 임베딩 벡터
    - **위치**: 입력 시퀀스의 시작 또는 중간에 삽입
    - **학습**: 태스크별 손실 함수로 최적화
    - **저장**: 작은 어댑터 파일로 저장
    """)
