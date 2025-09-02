.PHONY: help install install-dev test lint format clean build docker-build docker-run docker-dev

help: ## 도움말 표시
	@echo "PEFT Streamlit App - 사용 가능한 명령어:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 프로덕션 의존성 설치
	pip install -r requirements.txt

install-dev: ## 개발 의존성 설치
	pip install -r requirements-dev.txt
	pre-commit install

test: ## 테스트 실행
	pytest tests/ -v --cov=peft_utils --cov-report=html --cov-report=term-missing

test-fast: ## 빠른 테스트 실행 (커버리지 제외)
	pytest tests/ -v

lint: ## 코드 품질 검사
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	black --check --diff .
	isort --check-only --diff .

format: ## 코드 자동 포맷팅
	black .
	isort .

clean: ## 임시 파일 정리
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf .coverage

build: ## 패키지 빌드
	python -m build

docker-build: ## Docker 이미지 빌드
	docker build -t peft-streamlit .

docker-run: ## Docker 컨테이너 실행
	docker-compose up -d

docker-dev: ## 개발용 Docker 컨테이너 실행
	docker-compose --profile dev up -d

docker-stop: ## Docker 컨테이너 중지
	docker-compose down

docker-logs: ## Docker 로그 확인
	docker-compose logs -f

run: ## 로컬에서 Streamlit 앱 실행
	streamlit run app.py

run-dev: ## 개발 모드로 Streamlit 앱 실행
	streamlit run app.py --server.runOnSave=true

security-check: ## 보안 검사
	bandit -r . -f json -o bandit-report.json
	safety check --json --output safety-report.json

ci-check: ## CI 검사 (테스트 + 린트 + 보안)
	make test
	make lint
	make security-check

pre-commit-all: ## 모든 pre-commit 훅 실행
	pre-commit run --all-files

update-deps: ## 의존성 업데이트
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt

# 개발 환경 설정
setup-dev: install-dev ## 개발 환경 설정
	@echo "✅ 개발 환경 설정 완료!"
	@echo "📋 사용 가능한 명령어:"
	@echo "  make test      - 테스트 실행"
	@echo "  make lint      - 코드 품질 검사"
	@echo "  make format    - 코드 자동 포맷팅"
	@echo "  make run       - 앱 실행"
	@echo "  make docker-run - Docker로 실행"

# 프로덕션 환경 설정
setup-prod: install ## 프로덕션 환경 설정
	@echo "✅ 프로덕션 환경 설정 완료!"
	@echo "📋 사용 가능한 명령어:"
	@echo "  make run       - 앱 실행"
	@echo "  make docker-run - Docker로 실행"
	@echo "  make build     - 패키지 빌드"
