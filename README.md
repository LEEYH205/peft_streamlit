# 🧩 PEFT Hands-on Streamlit Application

초보자도 쉽게 이해할 수 있는 **Parameter-Efficient Fine-Tuning (PEFT)** 실습 애플리케이션입니다.

## 🚀 **즉시 체험하기**

### 🌐 **온라인 데모**
- **[Hugging Face Spaces](https://huggingface.co/spaces/LEEYH205/peft-hands-on)** 🚀
- **[Streamlit Cloud](https://peft-hands-on.streamlit.app)** ☁️

### 💻 **로컬 실행**
```bash
git clone https://github.com/LEEYH205/peft_streamlit.git
cd peft_streamlit
pip install -r requirements.txt
streamlit run app.py
```

## ✨ **지원하는 PEFT 방법**

| 방법 | 설명 | 파라미터 효율성 | 구현 상태 |
|------|------|----------------|-----------|
| **LoRA** | Low-Rank Adaptation | 0.8% | ✅ 완전 구현 |
| **DoRA** | Weight-Decomposed LoRA | 0.8% | ⚠️ LoRA로 시뮬레이션 |
| **AdaLoRA** | Adaptive LoRA | 0.8% | ⚠️ LoRA로 시뮬레이션 |
| **QLoRA** | 4-bit Quantized LoRA | 12.5% | ✅ 완전 구현 |
| **IA³** | Infused Adapter by Inhibiting and Amplifying Inner Activations | 0.04% | ✅ 완전 구현 |
| **Prefix Tuning** | Virtual Token Prefix | 0.1% | ✅ 완전 구현 |
| **Prompt Tuning** | Virtual Token Prompt | 0.05% | ✅ 완전 구현 |

## 🏗️ **프로젝트 구조**

```
peft-streamlit/
├── app.py                          # 메인 Streamlit 앱
├── pages/                          # PEFT 방법별 페이지
│   ├── 1_LoRA.py                  # LoRA 실습
│   ├── 2_DoRA.py                  # DoRA 실습
│   ├── 3_AdaLoRA.py              # AdaLoRA 실습
│   ├── 4_IA3.py                   # IA³ 실습
│   ├── 4_QLoRA.py                 # QLoRA 실습
│   ├── 5_Prefix_P_Tuning.py      # Prefix/Prompt Tuning 실습
│   └── 6_Evaluate_and_Compare.py # 성능 평가 및 비교
├── peft_utils/                     # 핵심 유틸리티
│   ├── data.py                     # 데이터 로딩
│   ├── model.py                    # 모델 관리
│   ├── train.py                    # 학습 로직
│   ├── eval.py                     # 평가 함수
│   └── viz.py                      # 시각화 및 한글 폰트
├── data/                           # 샘플 데이터
│   └── tiny_instruct.jsonl        # 작은 instruction 데이터
├── .streamlit/                     # Streamlit 설정
│   └── config.toml                # 배포 설정
└── tests/                          # 테스트 코드
```

## 🛠️ **기술 스택**

- **Frontend**: Streamlit 1.36.0+
- **ML Framework**: PyTorch 2.0.0+, Transformers 4.43.0+
- **PEFT**: PEFT 0.11.0+
- **Data**: Hugging Face Datasets 2.20.0+
- **Visualization**: Matplotlib 3.8.0+, Pandas 2.0.0+
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, isort, flake8

## 🚀 **배포 방법**

### 1. **Hugging Face Spaces 배포**

#### **자동 배포 (GitHub 연동)**
1. [Hugging Face Spaces](https://huggingface.co/spaces)에 접속
2. "Create new Space" 클릭
3. 설정:
   - **Owner**: 본인 계정
   - **Space name**: `peft-hands-on`
   - **Space SDK**: `Streamlit`
   - **License**: `MIT`
4. "Create Space" 클릭
5. GitHub 저장소와 연동:
   - **Repository**: `LEEYH205/peft_streamlit`
   - **Branch**: `main`
   - **Root directory**: `/` (루트)

#### **수동 배포**
```bash
# Hugging Face CLI 설치
pip install huggingface_hub

# 로그인
huggingface-cli login

# Space 생성
huggingface-cli repo create peft-hands-on --type space --space-sdk streamlit

# 코드 업로드
git clone https://huggingface.co/spaces/LEEYH205/peft-hands-on
cd peft-hands-on
# 파일 복사 후
git add .
git commit -m "Initial PEFT Hands-on app"
git push
```

#### **Hugging Face Spaces 설정 파일**
```yaml
# README.md (Hugging Face Spaces용)
title: PEFT Hands-on
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
```

### 2. **Streamlit Cloud 배포**

#### **GitHub 연동 배포**
1. [Streamlit Cloud](https://share.streamlit.io/)에 접속
2. "New app" 클릭
3. 설정:
   - **Repository**: `LEEYH205/peft_streamlit`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `peft-hands-on` (선택사항)
4. "Deploy!" 클릭

#### **배포 설정 파일**
```toml
# .streamlit/config.toml
[server]
headless = true
enableCORS = false
port = 8501
maxUploadSize = 200
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "info"
```

## ⚙️ **고급 설정**

### **환경 변수 설정**
```bash
# Hugging Face Spaces
SPACE_ID=your_space_id
HF_TOKEN=your_token

# Streamlit Cloud
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### **리소스 제한 설정**
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
enableXsrfProtection = false
enableCORS = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 📊 **성능 및 제한사항**

### **메모리 사용량**
- **로컬 실행**: 4-8GB RAM 권장
- **Hugging Face Spaces**: 16GB RAM (CPU), 32GB RAM (GPU)
- **Streamlit Cloud**: 1GB RAM (무료), 4GB RAM (Pro)

### **처리 속도**
- **CPU**: 기본 모델 로딩 2-5분, 학습 10-30분
- **GPU**: 기본 모델 로딩 30초-2분, 학습 2-10분

### **지원 환경**
- **OS**: Windows, macOS, Linux
- **Python**: 3.9, 3.10, 3.11, 3.12
- **브라우저**: Chrome, Firefox, Safari, Edge

## 🔧 **문제 해결**

### **자주 발생하는 문제**

#### 1. **모델 로딩 실패**
```bash
# 해결 방법
pip install --upgrade transformers
pip install --upgrade torch
```

#### 2. **한글 폰트 문제**
```python
# peft_utils/viz.py에서 자동 해결
setup_korean_font()  # 자동으로 적절한 폰트 선택
```

#### 3. **메모리 부족**
```python
# 더 작은 모델 사용
model_id = "microsoft/DialoGPT-small"  # 117M 파라미터
```

### **로그 확인**
```bash
# Streamlit 로그
streamlit run app.py --logger.level debug

# Hugging Face Spaces 로그
# Space 페이지의 "Logs" 탭 확인
```

## 🤝 **기여하기**

### **개발 환경 설정**
```bash
# 저장소 클론
git clone https://github.com/LEEYH205/peft_streamlit.git
cd peft_streamlit

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 개발 의존성 설치
pip install -r requirements-dev.txt

# Pre-commit 훅 설치
pre-commit install

# 테스트 실행
pytest tests/ -v
```

### **코드 품질**
```bash
# 포맷팅
black . --line-length=88
isort . --profile=black

# 린팅
flake8 . --max-line-length=88

# 보안 검사
bandit -r .
safety check
```

## 📈 **로드맵**

### **v1.1 (예정)**
- [ ] DoRA 완전 구현
- [ ] AdaLoRA 완전 구현
- [ ] 더 많은 모델 지원
- [ ] 성능 벤치마크 추가

### **v1.2 (예정)**
- [ ] 웹 UI 개선
- [ ] 실시간 학습 모니터링
- [ ] 모델 비교 도구
- [ ] 커스텀 데이터셋 지원

### **v1.3 (예정)**
- [ ] 분산 학습 지원
- [ ] 모델 압축 도구
- [ ] API 서버 제공
- [ ] 모바일 최적화

## 📄 **라이선스**

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

## 🙏 **감사의 말**

- **Hugging Face**: PEFT 라이브러리와 Spaces 플랫폼
- **Streamlit**: 멋진 웹 앱 프레임워크
- **AI 커뮤니티**: 지속적인 피드백과 제안

---

**⭐ 이 프로젝트가 도움이 되었다면 GitHub에 스타를 눌러주세요!**

**🚀 [Hugging Face Spaces](https://huggingface.co/spaces/LEEYH205/peft-hands-on)에서 바로 체험해보세요!**
**☁️ [Streamlit Cloud](https://peft-hands-on.streamlit.app)에서도 사용할 수 있습니다!**
