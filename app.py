import streamlit as st
from peft_utils.data import load_tiny_instruct
from peft_utils.viz import setup_korean_font

# 한글 폰트 전역 설정
setup_korean_font()

st.set_page_config(page_title="PEFT Hands-on", page_icon="🧩", layout="wide")
st.title("🧩 PEFT Hands-on (LoRA / QLoRA / IA³ / Prefix / Prompt)")

st.markdown(
"""
초보자도 **PEFT(파라미터 효율적 미세조정)**를 쉽게 이해하고 실습할 수 있도록 만든 대화형 튜토리얼입니다.
좌측 페이지에서 기법을 선택하고, 기본 하이퍼파라미터로 **빠른 데모 학습**을 실행해보세요.
"""
)

train_ds, eval_ds = load_tiny_instruct()
st.subheader("📦 샘플 데이터 (tiny_instruct)")
st.dataframe(train_ds.to_pandas().head(3))
st.info("좌측 사이드바의 페이지를 눌러 각 기법(LoRA, QLoRA, IA³, Prefix/Prompt)을 실습하세요.")
