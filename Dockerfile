# Docker �̹����� ���̽��� ���� Python �̹����� �����մϴ�. 
FROM python:3.11

# �۾� ���丮�� �����մϴ�.
WORKDIR /app

# ������ ����� ���� requirements.txt�� Docker �����̳� ������ �����մϴ�.
COPY requirements.txt .

# pip�� ����� �������� ��ġ�մϴ�.
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

# ���ø����̼��� ��� ������ Docker �����̳� ������ �����մϴ�.
COPY . .

# Docker �����̳ʰ� ���۵� �� ����� ��ɾ �����մϴ�.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]