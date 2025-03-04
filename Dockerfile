# 베이스 이미지: python:3.9-slim
FROM python:3.9-slim

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 리눅스 패키지 설치
# libreoffice, unoconv, 폰트 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    unoconv \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 로컬 소스 모두 복사 (make_report.py, make_pdf.py, config.ini, reports_image 폴더 등등)
COPY . /app

# pip 최신 버전으로 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# setup.sh 등에서 설치하던 Python 라이브러리를 바로 설치
RUN pip install --no-cache-dir \
    python-docx \
    fastapi \
    pydantic \
    uvicorn \
    numpy \
    pandas \
    scikit-learn \
    openai==0.28

# Host와 Port 지정
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "make_report:app", "--host", "0.0.0.0", "--port", "8000"]