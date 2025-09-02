"""
PEFT Hands-on Streamlit Application

초보자도 쉽게 이해할 수 있는 Parameter-Efficient Fine-Tuning (PEFT) 실습 애플리케이션입니다.
LoRA, QLoRA, IA³, Prefix Tuning, Prompt Tuning 등 다양한 PEFT 방법을 실습할 수 있습니다.

🚀 배포 플랫폼:
- Hugging Face Spaces
- Streamlit Cloud
- 로컬 실행
"""

import os

import streamlit as st

from peft_utils.data import load_tiny_instruct
from peft_utils.viz import setup_korean_font

# 환경 변수 설정 (배포 환경 감지)
IS_HF_SPACES = os.getenv("SPACE_ID") is not None
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SERVER_PORT") is not None

# 한글 폰트 전역 설정
setup_korean_font()

# 페이지 설정
st.set_page_config(
    page_title="PEFT Hands-on",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 배포 환경 표시
if IS_HF_SPACES:
    st.sidebar.success("🚀 **Hugging Face Spaces**에서 실행 중")
elif IS_STREAMLIT_CLOUD:
    st.sidebar.success("☁️ **Streamlit Cloud**에서 실행 중")
else:
    st.sidebar.info("💻 **로컬 환경**에서 실행 중")

st.title("🧩 PEFT Hands-on (LoRA / QLoRA / IA³ / Prefix / Prompt)")

# 배포 환경별 설명
if IS_HF_SPACES or IS_STREAMLIT_CLOUD:
    st.info(
        """
    🌟 **클라우드 배포 완료!**

    이제 인터넷이 연결된 어디서든 PEFT를 학습할 수 있습니다.
    - **Hugging Face Spaces**: AI 커뮤니티와 공유
    - **Streamlit Cloud**: 안정적인 클라우드 서비스
    """
    )

st.markdown(
    """
초보자도 **PEFT(파라미터 효율적 미세조정)**를 쉽게 이해하고 실습할 수 있도록 만든 대화형 튜토리얼입니다.
좌측 페이지에서 기법을 선택하고, 기본 하이퍼파라미터로 **빠른 데모 학습**을 실행해보세요.
"""
)

# 데이터 로드 (배포 환경에서는 에러 처리 강화)
try:
    train_ds, eval_ds = load_tiny_instruct()
    st.subheader("📦 샘플 데이터 (tiny_instruct)")
    st.dataframe(train_ds.to_pandas().head(3))
    st.info(
        "좌측 사이드바의 페이지를 눌러 각 기법(LoRA, QLoRA, IA³, Prefix/Prompt)을 실습하세요."
    )
except Exception as e:
    st.warning(f"⚠️ 데이터 로드 중 오류가 발생했습니다: {str(e)}")
    st.info(
        """
    💡 **해결 방법**:
    1. 페이지를 새로고침해보세요
    2. 잠시 후 다시 시도해보세요
    3. 문제가 지속되면 GitHub 이슈를 등록해주세요
    """
    )

# 배포 정보 표시
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 **프로젝트 정보**")
    st.markdown(
        "**GitHub**: [peft_streamlit](https://github.com/LEEYH205/peft_streamlit)"
    )

    if IS_HF_SPACES:
        st.markdown(
            "**Hugging Face**: [PEFT Hands-on](https://huggingface.co/spaces/LEEYH205/peft-hands-on)"
        )

    if IS_STREAMLIT_CLOUD:
        st.markdown(
            "**Streamlit Cloud**: [PEFT-Hands-on-lyh205](https://peft-hands-on-lyh205.streamlit.app)"
        )

    st.markdown("**버전**: 1.0.0")
    st.markdown("**라이선스**: MIT")
