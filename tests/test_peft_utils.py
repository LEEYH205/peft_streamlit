"""
PEFT Utils 테스트 모듈
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# 테스트할 모듈들 import
try:
    from peft_utils.data import load_tiny_instruct
    from peft_utils.eval import perplexity
    from peft_utils.model import DEFAULT_MODEL_ID, load_base_model
    from peft_utils.train import build_adapter, train_once
    from peft_utils.viz import create_comparison_chart, setup_korean_font

    PEFT_UTILS_AVAILABLE = True
except ImportError:
    PEFT_UTILS_AVAILABLE = False


@pytest.mark.skipif(not PEFT_UTILS_AVAILABLE, reason="PEFT utils not available")
class TestPEFTUtils:
    """PEFT 유틸리티 테스트 클래스"""

    def test_setup_korean_font(self):
        """한글 폰트 설정 테스트"""
        try:
            setup_korean_font()
            # 폰트 설정이 성공적으로 완료되었는지 확인
            assert True
        except Exception as e:
            pytest.fail(f"한글 폰트 설정 실패: {e}")

    def test_create_comparison_chart(self):
        """비교 차트 생성 테스트"""
        methods = ["LoRA", "DoRA", "AdaLoRA"]
        values = [0.8, 0.8, 0.1]
        title = "파라미터 효율성 비교"
        ylabel = "파라미터 사용률 (%)"

        try:
            fig = create_comparison_chart(methods, values, title, ylabel)
            assert fig is not None
            # matplotlib figure 객체인지 확인
            assert hasattr(fig, "savefig")
        except Exception as e:
            pytest.fail(f"비교 차트 생성 실패: {e}")

    def test_load_tiny_instruct(self):
        """Tiny Instruct 데이터 로드 테스트"""
        try:
            # 데이터 로드 테스트 (작은 샘플)
            data = load_tiny_instruct(max_samples=2)
            assert len(data) <= 2
            if len(data) > 0:
                assert "text" in data[0]
        except Exception as e:
            pytest.fail(f"Tiny Instruct 데이터 로드 실패: {e}")

    @patch("peft_utils.model.AutoModelForCausalLM.from_pretrained")
    @patch("peft_utils.model.AutoTokenizer.from_pretrained")
    def test_load_base_model(self, mock_tokenizer, mock_model):
        """베이스 모델 로드 테스트 (모킹)"""
        # Mock 설정
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()

        try:
            model, tokenizer = load_base_model()
            assert model is not None
            assert tokenizer is not None
        except Exception as e:
            pytest.fail(f"베이스 모델 로드 실패: {e}")

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

    def test_perplexity_single_text(self):
        """단일 텍스트 perplexity 계산 테스트"""
        # Mock 모델과 토크나이저
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.parameters.return_value = [torch.randn(10, 10)]

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "test text"

        try:
            # 단일 텍스트로 perplexity 계산
            ppl = perplexity(mock_model, mock_tokenizer, "테스트 텍스트")
            assert isinstance(ppl, (int, float)) or ppl is None
        except Exception as e:
            pytest.fail(f"단일 텍스트 perplexity 계산 실패: {e}")


class TestStreamlitApp:
    """Streamlit 앱 테스트 클래스"""

    def test_app_import(self):
        """앱 모듈 import 테스트"""
        try:
            import app

            assert True
        except ImportError as e:
            pytest.fail(f"앱 모듈 import 실패: {e}")

    def test_pages_import(self):
        """페이지 모듈들 import 테스트"""
        try:
            # 각 페이지 모듈 import 테스트 (동적 import 사용)
            import importlib

            import pages

            # 페이지 모듈들이 존재하는지 확인
            assert hasattr(pages, "1_LoRA")
            assert hasattr(pages, "2_DoRA")
            assert hasattr(pages, "3_AdaLoRA")
            assert hasattr(pages, "4_IA3")
            assert hasattr(pages, "4_QLoRA")
            assert hasattr(pages, "5_Prefix_P_Tuning")
            assert hasattr(pages, "6_Evaluate_and_Compare")
            assert True
        except ImportError as e:
            pytest.fail(f"페이지 모듈 import 실패: {e}")


class TestRequirements:
    """요구사항 테스트 클래스"""

    def test_requirements_file_exists(self):
        """requirements.txt 파일 존재 확인"""
        import os

        assert os.path.exists(
            "requirements.txt"
        ), "requirements.txt 파일이 존재하지 않습니다"

    def test_requirements_content(self):
        """requirements.txt 내용 검증"""
        with open("requirements.txt", "r") as f:
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
