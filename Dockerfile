# Docker 이미지의 베이스로 사용될 Python 이미지를 선택합니다. 
FROM python:3.11

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 의존성 목록을 가진 requirements.txt를 Docker 컨테이너 안으로 복사합니다.
COPY requirements.txt .

# pip를 사용해 의존성을 설치합니다.
RUN pip install --upgrade pip

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

# Installing mecab for linux 
RUN apt-get update && apt-get install -y \
    curl \
    git \
    g++ \
    openjdk-8-jdk \
    python3-dev \
    python3-pip \
    build-essential

RUN bash -c "$(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)"

# 애플리케이션의 모든 파일을 Docker 컨테이너 안으로 복사합니다.
COPY . .

# Docker 컨테이너가 시작될 때 실행될 명령어를 설정합니다.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]