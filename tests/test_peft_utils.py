"""
PEFT Utils 테스트 모듈
"""

import os
from unittest.mock import Mock

import pytest

# 상대 경로로 import (sys.path.insert 없이)
try:
    from peft_utils.data import load_tiny_instruct
    from peft_utils.eval import perplexity
    from peft_utils.train import build_adapter
    from peft_utils.viz import setup_korean_font

    PEFT_UTILS_AVAILABLE = True
except ImportError:
    PEFT_UTILS_AVAILABLE = False


@pytest.mark.skipif(not PEFT_UTILS_AVAILABLE, reason="PEFT utils not available")
class TestDataUtils:
    """데이터 유틸리티 테스트 클래스"""

    def test_load_tiny_instruct(self):
        """tiny_instruct 데이터 로드 테스트"""
        try:
            train_ds, eval_ds = load_tiny_instruct()
            assert train_ds is not None
            assert eval_ds is not None
        except Exception as e:
            pytest.fail(f"tiny_instruct 데이터 로드 실패: {e}")


@pytest.mark.skipif(not PEFT_UTILS_AVAILABLE, reason="PEFT utils not available")
class TestVizUtils:
    """시각화 유틸리티 테스트 클래스"""

    def test_setup_korean_font(self):
        """한글 폰트 설정 테스트"""
        try:
            setup_korean_font()
            assert True
        except Exception as e:
            pytest.fail(f"한글 폰트 설정 실패: {e}")


@pytest.mark.skipif(not PEFT_UTILS_AVAILABLE, reason="PEFT utils not available")
class TestModelUtils:
    """모델 유틸리티 테스트 클래스"""

    def test_build_adapter_lora(self):
        """LoRA 어댑터 빌드 테스트"""
        # Mock 모델 생성
        mock_model = Mock()
        mock_model.config.hidden_size = 768

        try:
            adapter = build_adapter(mock_model, method="lora", r=8, alpha=16)
            assert adapter is not None
        except Exception as e:
            pytest.fail(f"LoRA 어댑터 빌드 실패: {e}")

    def test_build_adapter_ia3(self):
        """IA³ 어댑터 빌드 테스트"""
        # Mock 모델 생성
        mock_model = Mock()
        mock_model.config.hidden_size = 768

        try:
            adapter = build_adapter(mock_model, method="ia3")
            assert adapter is not None
        except Exception as e:
            pytest.fail(f"IA³ 어댑터 빌드 실패: {e}")


class TestStreamlitApp:
    """Streamlit 앱 테스트 클래스"""

    def test_app_file_exists(self):
        """앱 파일 존재 확인"""
        app_file = "app.py"
        assert os.path.exists(app_file), "app.py 파일이 존재하지 않습니다"

    def test_pages_directory_exists(self):
        """pages 디렉토리 존재 확인"""
        pages_dir = "pages"
        assert os.path.isdir(pages_dir), "pages 디렉토리가 존재하지 않습니다"

    def test_pages_files_exist(self):
        """페이지 파일들 존재 확인"""
        pages_dir = "pages"
        expected_files = [
            "1_LoRA.py",
            "2_DoRA.py",
            "3_AdaLoRA.py",
            "4_IA3.py",
            "4_QLoRA.py",
            "5_Prefix_P_Tuning.py",
            "6_Evaluate_and_Compare.py",
        ]

        for file_name in expected_files:
            file_path = os.path.join(pages_dir, file_name)
            assert os.path.exists(file_path), f"{file_name} 파일이 존재하지 않습니다"


class TestRequirements:
    """요구사항 테스트 클래스"""

    def test_requirements_file_exists(self):
        """requirements.txt 파일 존재 확인"""
        requirements_file = "requirements.txt"
        assert os.path.exists(
            requirements_file
        ), "requirements.txt 파일이 존재하지 않습니다"

    def test_requirements_content(self):
        """requirements.txt 내용 검증"""
        requirements_file = "requirements.txt"
        with open(requirements_file, "r") as f:
            content = f.read()
            # 필수 패키지들이 포함되어 있는지 확인
            assert "streamlit" in content, "streamlit이 requirements.txt에 없습니다"
            assert "torch" in content, "torch가 requirements.txt에 없습니다"
            assert (
                "transformers" in content
            ), "transformers가 requirements.txt에 없습니다"
            assert "peft" in content, "peft가 requirements.txt에 없습니다"


if __name__ == "__main__":
    pytest.main([__file__])
