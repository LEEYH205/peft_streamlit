import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import numpy as np
import warnings

def setup_korean_font():
    """한글 폰트 설정 - 전역 적용"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # macOS에서 사용 가능한 한글 폰트들
        korean_fonts = ['AppleGothic', 'Arial Unicode MS', 'Hiragino Sans', 'PingFang SC', 'STHeiti']
    elif system == "Windows":
        # Windows에서 사용 가능한 한글 폰트들
        korean_fonts = ['Malgun Gothic', 'Gulim', 'Batang', 'Dotum', 'Arial Unicode MS']
    else:  # Linux
        # Linux에서 사용 가능한 한글 폰트들
        korean_fonts = ['NanumGothic', 'DejaVu Sans', 'Liberation Sans', 'Noto Sans CJK KR']
    
    # 사용 가능한 폰트 찾기
    available_fonts = []
    for font in korean_fonts:
        try:
            fm.findfont(font)
            available_fonts.append(font)
        except:
            continue
    
    # 폰트 설정
    if available_fonts:
        plt.rcParams['font.family'] = available_fonts[0]
        # 추가 폰트 설정
        plt.rcParams['font.sans-serif'] = [available_fonts[0]] + plt.rcParams['font.sans-serif']
        print(f"한글 폰트 설정 완료: {available_fonts[0]}")
    else:
        # 폰트가 없으면 기본 폰트 사용하고 경고
        print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 음수 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
    # 한글 경고 무시
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # 폰트 캐시 초기화 (안전한 방법)
    try:
        # matplotlib 3.6+ 버전
        if hasattr(fm, '_rebuild'):
            fm._rebuild()
        elif hasattr(fm, 'fontManager'):
            fm.fontManager.cachedir = None
    except:
        # 폰트 캐시 초기화 실패 시 무시
        pass

def create_comparison_chart(methods, values, title, ylabel, colors=None):
    """비교 차트 생성 - 한글 지원 강화"""
    setup_korean_font()  # 매번 폰트 설정 확인
    
    if colors is None:
        colors = ['red', 'lightgreen', 'green', 'darkgreen', 'blue', 'purple']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # 값 표시
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if isinstance(value, (int, float)):
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_structure_diagram(title, elements, arrows, figsize=(6, 4)):
    """구조도 생성 - 한글 지원 강화"""
    setup_korean_font()  # 매번 폰트 설정 확인
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 요소 배치
    for i, (text, pos, color) in enumerate(elements):
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
    
    # 화살표 그리기
    for start, end, color in arrows:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], 
                head_width=0.05, head_length=0.05, fc=color, ec=color)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def create_korean_chart(title, xlabel, ylabel, figsize=(8, 6)):
    """한글 지원 차트 생성 - 기본 설정"""
    setup_korean_font()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax
