# 🧩 PEFT Hands-on Streamlit App

**초보자도 쉽게 이해할 수 있는 Parameter-Efficient Fine-Tuning (PEFT) 실습 애플리케이션**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PEFT](https://img.shields.io/badge/PEFT-0.7.0+-green.svg)](https://github.com/huggingface/peft)
[![Transformers](https://img.shields.io/badge/Transformers-4.35.0+-orange.svg)](https://huggingface.co/docs/transformers/)

## 🎯 프로젝트 소개

이 프로젝트는 **PEFT (Parameter-Efficient Fine-Tuning)** 기법들을 초보자도 쉽게 이해하고 실습할 수 있도록 만든 **Streamlit 웹 애플리케이션**입니다.

### ✨ 주요 특징

- 🚀 **7가지 PEFT 방법** 완벽 지원
- 🎨 **초등학생도 이해할 수 있는** 상세한 설명과 시각화
- 🔧 **실제 구현 가이드** 제공
- 🌏 **한글 완벽 지원** (그래프, 설명, UI)
- 💻 **Mac/Windows/Linux** 크로스 플랫폼 지원
- 🎯 **실습 가능한 학습 환경** 제공

## 🏗️ 지원하는 PEFT 방법

| 방법 | 상태 | 설명 | 파라미터 효율성 |
|------|------|------|----------------|
| **LoRA** | ✅ 완벽 지원 | 저랭크 적응 (Low-Rank Adaptation) | 0.8% |
| **DoRA** | 🔄 동적 지원 | 가중치 분해 LoRA (Weight-Decomposed LoRA) | 0.8%+ |
| **AdaLoRA** | 🔄 동적 지원 | 적응형 LoRA (Adaptive Low-Rank Adaptation) | 0.1-0.8% |
| **QLoRA** | ✅ 완벽 지원 | 4-bit 양자화 + LoRA | 0.8% |
| **IA³** | ✅ 완벽 지원 | 스케일링 벡터 학습 | 0.04% |
| **Prefix Tuning** | ✅ 완벽 지원 | 가상 토큰 학습 | 0.1% |
| **Prompt Tuning** | ✅ 완벽 지원 | 프롬프트 학습 | 0.1% |

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone <repository-url>
cd peft-streamlit
```

### 2. 가상환경 생성 및 활성화
```bash
# Python 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. Streamlit 앱 실행
```bash
streamlit run app.py
```

### 5. 브라우저에서 접속
```
http://localhost:8501
```

## 📱 앱 구조

### 🏠 메인 페이지
- **PEFT 개요**: 모든 PEFT 방법의 간단한 소개
- **성능 비교**: 각 방법의 장단점과 사용 시나리오
- **시작 가이드**: 초보자를 위한 단계별 가이드

### 📚 PEFT 방법별 페이지

#### 1. **LoRA** - 저랭크 적응
- 📊 **수학적 원리**: W = W₀ + A×B
- 🎨 **시각화**: 가중치 구조, 학습 과정, 파라미터 효율성
- 🔧 **실습**: 실제 LoRA 학습 및 평가
- 💡 **활용 팁**: rank, alpha 설정 가이드

#### 2. **DoRA** - 가중치 분해 LoRA
- 📊 **핵심 아이디어**: 크기(magnitude)와 방향(direction) 분리
- 🎨 **구조 비교**: LoRA vs DoRA 시각적 비교
- 🔧 **실제 구현**: DoRAConfig 사용법 (지원 시)
- 🔄 **시뮬레이션**: LoRA로 DoRA 효과 시뮬레이션

#### 3. **AdaLoRA** - 적응형 LoRA
- 📊 **동적 rank**: 학습 중 자동 rank 조정
- 🎨 **중요도 평가**: 레이어별 중요도 시각화
- 🔧 **실제 구현**: AdaLoRAConfig 사용법 (지원 시)
- 🔄 **시뮬레이션**: LoRA로 AdaLoRA 효과 시뮬레이션

#### 4. **QLoRA** - 4-bit 양자화 + LoRA
- 📊 **메모리 효율성**: 4-bit 양자화로 메모리 절약
- 🎨 **양자화 과정**: 가중치 압축 시각화
- 🔧 **Mac 환경 대응**: 4-bit 제한사항 설명 및 대안
- 💾 **메모리 vs 성능**: 트레이드오프 분석

#### 5. **IA³** - 스케일링 벡터 학습
- 📊 **핵심 원리**: W = W₀ ⊙ s (요소별 곱셈)
- 🎨 **구조 비교**: LoRA vs IA³ 시각적 비교
- 🔧 **실습**: IA³ 학습 및 스케일링 벡터 효과
- ⚡ **초고효율**: 0.04% 파라미터 사용

#### 6. **Prefix & Prompt Tuning** - 가상 토큰 학습
- 📊 **가상 토큰**: 학습 가능한 프롬프트/접두사
- 🎨 **학습 과정**: 가상 토큰 변화 시각화
- 🔧 **실습**: Prefix/Prompt Tuning 학습
- 🎯 **사용법**: 텍스트 생성 및 조작

#### 7. **평가 및 비교** - 성능 분석
- 📊 **성능 지표**: Perplexity, 학습 속도, 메모리 사용량
- 🎨 **비교 차트**: 모든 PEFT 방법의 종합 비교
- 🔧 **실제 평가**: 사용자 입력 텍스트로 성능 테스트
- 📋 **선택 가이드**: 상황별 최적 PEFT 방법 추천

## 🛠️ 기술 스택

### 백엔드
- **Python 3.8+**: 메인 프로그래밍 언어
- **Streamlit**: 웹 애플리케이션 프레임워크
- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 트랜스포머 라이브러리
- **PEFT**: Parameter-Efficient Fine-Tuning 라이브러리

### 프론트엔드
- **Streamlit Components**: 대화형 UI 컴포넌트
- **Matplotlib**: 그래프 및 시각화
- **Plotly**: 인터랙티브 차트
- **Pandas**: 데이터 처리 및 표시

### 데이터
- **Tiny Instruct**: 경량 명령어 데이터셋
- **Hugging Face Datasets**: 데이터 로딩 및 처리

## 🔧 고급 설정

### PEFT 버전 호환성
```bash
# 최신 PEFT 버전 설치 (DoRA, AdaLoRA 지원)
pip install --upgrade peft

# 개발 버전 설치
pip install git+https://github.com/huggingface/peft.git
```

### GPU 가속 설정
```bash
# CUDA 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# MPS (Apple Silicon) 지원
# 자동으로 감지되어 사용됨
```

### 한글 폰트 설정
```python
# 자동으로 시스템에 맞는 한글 폰트 설정
from peft_utils.viz import setup_korean_font
setup_korean_font()
```

## 📊 성능 및 제한사항

### ✅ 지원 환경
- **OS**: macOS, Windows, Linux
- **Python**: 3.8, 3.9, 3.10, 3.11, 3.12
- **하드웨어**: CPU, GPU (CUDA), Apple Silicon (MPS)
- **메모리**: 최소 4GB RAM 권장

### ⚠️ 제한사항
- **Mac 환경**: 4-bit 양자화 제한 (QLoRA에서 일반 LoRA로 폴백)
- **PEFT 버전**: 일부 최신 기능은 PEFT 0.7.0+ 필요
- **모델 크기**: 대용량 모델은 메모리 제한으로 제한될 수 있음

### 🔄 자동 대체
- **DoRA/AdaLoRA 미지원**: LoRA로 자동 시뮬레이션
- **Prefix/Prompt Tuning 실패**: LoRA로 자동 대체
- **4-bit 양자화 실패**: 일반 모델로 자동 폴백

## 🎓 학습 가이드

### 👶 초보자 (Beginner)
1. **메인 페이지**에서 PEFT 개요 파악
2. **LoRA 페이지**부터 시작하여 기본 개념 이해
3. **시각화**를 통해 수학적 원리 직관적 파악
4. **실습**으로 실제 학습 과정 체험

### 🧑‍🎓 중급자 (Intermediate)
1. **DoRA, AdaLoRA**로 고급 LoRA 기법 학습
2. **IA³**로 초고효율 PEFT 방법 이해
3. **QLoRA**로 메모리 최적화 기법 습득
4. **평가 및 비교**로 성능 분석 능력 향상

### 🧑‍💻 고급자 (Advanced)
1. **실제 구현** 가이드를 통한 프로덕션 적용
2. **커스텀 설정**으로 최적화된 PEFT 구현
3. **성능 튜닝**을 통한 효율성 극대화
4. **새로운 PEFT 방법** 연구 및 실험

## 🐛 문제 해결

### 일반적인 문제들

#### 1. **Import 에러**
```bash
# PEFT 라이브러리 재설치
pip uninstall peft
pip install peft
```

#### 2. **한글 폰트 문제**
```python
# 한글 폰트 자동 설정
from peft_utils.viz import setup_korean_font
setup_korean_font()
```

#### 3. **메모리 부족**
```python
# 배치 크기 줄이기
batch_size = 1  # 기본값: 2
```

#### 4. **학습 실패**
```python
# 에러 로그 확인 후 자동 대체 방법 사용
# Prefix/Prompt Tuning 실패 시 LoRA로 자동 전환
```

### 디버깅 모드
```bash
# 상세한 로그와 함께 실행
streamlit run app.py --logger.level debug
```

## 🤝 기여하기

### 버그 리포트
- **GitHub Issues**에 상세한 에러 로그와 함께 리포트
- **재현 단계**를 명확하게 기술
- **환경 정보** (OS, Python 버전, 라이브러리 버전) 포함

### 기능 제안
- **Feature Request** 이슈로 새로운 기능 제안
- **사용 사례**와 **기대 효과** 명시
- **구현 아이디어** 제시

### 코드 기여
1. **Fork** 저장소
2. **Feature branch** 생성
3. **코드 작성** 및 **테스트**
4. **Pull Request** 생성

## 📄 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다.

## 🙏 감사의 말

- **Hugging Face**: PEFT 및 Transformers 라이브러리
- **Streamlit**: 웹 애플리케이션 프레임워크
- **PyTorch**: 딥러닝 프레임워크
- **커뮤니티**: 피드백과 기여

## 📞 연락처

- **GitHub**: [프로젝트 저장소](https://github.com/your-username/peft-streamlit)
- **이슈**: [GitHub Issues](https://github.com/your-username/peft-streamlit/issues)
- **문의**: [이메일 또는 기타 연락처]

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

**🚀 PEFT의 세계로 떠나보세요!**
