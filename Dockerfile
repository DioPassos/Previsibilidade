FROM python:3.11-slim

WORKDIR /Previsibilidade

COPY requirements.txt .

RUN pip install --no-cache-dir --no-compile -r requirements.txt

COPY . .

CMD ["python", "-m", "collector.collect_pix_data"]