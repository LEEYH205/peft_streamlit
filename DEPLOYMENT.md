# 🚀 **배포 가이드 - Hugging Face Spaces & Streamlit Cloud**

PEFT Hands-on Streamlit 앱을 클라우드에 배포하는 방법을 단계별로 안내합니다.

## 🌟 **배포 옵션 비교**

| 플랫폼 | 장점 | 단점 | 무료 티어 | 적합한 사용자 |
|--------|------|------|-----------|---------------|
| **Hugging Face Spaces** | AI 커뮤니티 노출, GPU 지원 | 설정 복잡, 제한적 | ✅ 16GB RAM | AI 연구자, 커뮤니티 공유 |
| **Streamlit Cloud** | 간단한 설정, 안정적 | GPU 제한, 커스터마이징 제한 | ✅ 1GB RAM | 개발자, 빠른 배포 |

## 🚀 **1. Hugging Face Spaces 배포**

### **1-1. 자동 배포 (GitHub 연동 - 권장)**

#### **단계 1: Hugging Face 계정 생성**
1. [Hugging Face](https://huggingface.co/)에 접속
2. "Sign Up" 클릭하여 계정 생성
3. 이메일 인증 완료

#### **단계 2: Space 생성**
1. [Hugging Face Spaces](https://huggingface.co/spaces)에 접속
2. "Create new Space" 버튼 클릭
3. 다음 설정으로 Space 생성:

```yaml
Owner: [본인 계정명]
Space name: peft-hands-on
Space SDK: Streamlit
License: MIT
```

#### **단계 3: GitHub 저장소 연동**
1. 생성된 Space에서 "Settings" 탭 클릭
2. "Repository" 섹션에서:
   - **Repository**: `LEEYH205/peft_streamlit`
   - **Branch**: `main`
   - **Root directory**: `/` (루트)
3. "Save" 클릭

#### **단계 4: 자동 배포 확인**
- GitHub에 코드를 푸시하면 자동으로 배포됨
- Space 페이지에서 "Logs" 탭으로 배포 상태 확인

### **1-2. 수동 배포**

#### **단계 1: Hugging Face CLI 설치**
```bash
pip install huggingface_hub
```

#### **단계 2: 로그인**
```bash
huggingface-cli login
# 토큰 입력 (Hugging Face 설정에서 생성)
```

#### **단계 3: Space 생성**
```bash
huggingface-cli repo create peft-hands-on --type space --space-sdk streamlit
```

#### **단계 4: 코드 업로드**
```bash
# Space 클론
git clone https://huggingface.co/spaces/[계정명]/peft-hands-on
cd peft-hands-on

# 프로젝트 파일 복사
cp -r /path/to/peft-streamlit/* .

# 배포
git add .
git commit -m "Initial PEFT Hands-on app"
git push
```

### **1-3. Hugging Face Spaces 설정 파일**

#### **README.md (Space 전용)**
```yaml
---
title: PEFT Hands-on
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
---
```

#### **requirements.txt 확인**
- 모든 의존성이 포함되어 있는지 확인
- 버전 충돌이 없는지 확인

## ☁️ **2. Streamlit Cloud 배포**

### **2-1. GitHub 연동 배포 (권장)**

#### **단계 1: Streamlit Cloud 계정 생성**
1. [Streamlit Cloud](https://share.streamlit.io/)에 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭

#### **단계 2: 앱 설정**
```yaml
Repository: LEEYH205/peft_streamlit
Branch: main
Main file path: app.py
App URL: peft-hands-on (선택사항)
```

#### **단계 3: 배포 실행**
1. "Deploy!" 버튼 클릭
2. 배포 진행 상황 모니터링
3. 배포 완료 후 제공된 URL로 접속

### **2-2. 수동 배포**

#### **단계 1: Streamlit 앱 패키징**
```bash
# 프로젝트 디렉토리에서
pip install streamlit
streamlit run app.py --server.headless true
```

#### **단계 2: 배포 파일 생성**
```bash
# requirements.txt 확인
pip freeze > requirements.txt

# .streamlit/config.toml 확인
mkdir -p .streamlit
# config.toml 파일 생성 (이미 있음)
```

## ⚙️ **3. 배포 설정 최적화**

### **3-1. Streamlit 설정 최적화**

#### **`.streamlit/config.toml`**
```toml
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

### **3-2. 환경 변수 설정**

#### **Hugging Face Spaces**
```bash
SPACE_ID=your_space_id
HF_TOKEN=your_token
```

#### **Streamlit Cloud**
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### **3-3. 리소스 제한 설정**

#### **Hugging Face Spaces**
- **CPU**: 16GB RAM
- **GPU**: 32GB RAM (T4, V100 등)

#### **Streamlit Cloud**
- **무료**: 1GB RAM
- **Pro**: 4GB RAM

## 🔧 **4. 배포 후 문제 해결**

### **4-1. 일반적인 배포 문제**

#### **Import 오류**
```bash
# requirements.txt에 누락된 패키지 추가
pip install missing_package
pip freeze > requirements.txt
```

#### **메모리 부족**
```python
# 더 작은 모델 사용
model_id = "microsoft/DialoGPT-small"  # 117M 파라미터
```

#### **한글 폰트 문제**
```python
# peft_utils/viz.py에서 자동 해결
setup_korean_font()  # 자동으로 적절한 폰트 선택
```

### **4-2. 로그 확인**

#### **Hugging Face Spaces**
1. Space 페이지에서 "Logs" 탭 클릭
2. 실시간 로그 확인
3. 오류 메시지 분석

#### **Streamlit Cloud**
1. 앱 페이지에서 "Manage app" 클릭
2. "Logs" 섹션에서 로그 확인
3. 오류 메시지 분석

### **4-3. 성능 최적화**

#### **모델 캐싱**
```python
@st.cache_resource
def load_model(model_id):
    # 모델 로딩 로직
    pass
```

#### **데이터 캐싱**
```python
@st.cache_data
def load_dataset():
    # 데이터셋 로딩 로직
    pass
```

## 📊 **5. 모니터링 및 유지보수**

### **5-1. 성능 모니터링**

#### **메모리 사용량**
- 정기적으로 메모리 사용량 확인
- 필요시 모델 크기 조정

#### **응답 시간**
- 사용자 피드백 수집
- 성능 병목 지점 파악

### **5-2. 업데이트 및 배포**

#### **자동 배포 (GitHub 연동)**
```bash
# 로컬에서 코드 수정
git add .
git commit -m "Update app"
git push origin main
# 자동으로 배포됨
```

#### **수동 배포**
```bash
# Hugging Face Spaces
cd peft-hands-on
git pull origin main
git push

# Streamlit Cloud
# GitHub에 푸시하면 자동 배포
```

## 🌟 **6. 배포 완료 후**

### **6-1. 공유 및 홍보**

#### **GitHub README 업데이트**
```markdown
## 🚀 **즉시 체험하기**

- **[Hugging Face Spaces](https://huggingface.co/spaces/[계정명]/peft-hands-on)** 🚀
- **[Streamlit Cloud](https://[앱명].streamlit.app)** ☁️
```

#### **커뮤니티 공유**
- AI 관련 포럼에 공유
- GitHub에 스타 요청
- 피드백 수집

### **6-2. 지속적 개선**

#### **사용자 피드백 반영**
- 오류 리포트 수집
- 기능 요청 분석
- UI/UX 개선

#### **성능 최적화**
- 로딩 속도 개선
- 메모리 사용량 최적화
- 새로운 PEFT 방법 추가

## 🔗 **7. 유용한 링크**

- **[Hugging Face Spaces](https://huggingface.co/spaces)**: AI 앱 배포 플랫폼
- **[Streamlit Cloud](https://share.streamlit.io/)**: Streamlit 앱 호스팅
- **[GitHub Actions](https://github.com/features/actions)**: CI/CD 자동화
- **[PEFT Documentation](https://huggingface.co/docs/peft)**: PEFT 라이브러리 문서

---

**🚀 이제 PEFT Hands-on 앱을 클라우드에 배포하여 전 세계 사람들과 공유하세요!**

**💡 문제가 발생하면 GitHub Issues에 등록하거나 커뮤니티에 문의해주세요.**
