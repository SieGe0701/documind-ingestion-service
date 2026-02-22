FROM python:3.11-slim@sha256:6f6c5667f0dac8b5e2b3e8b8e8b3e8b8e8b3e8b8e8b3e8b8e8b3e8b8e8b3e8b

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN set -eux; \
	grep -vE '^(pytest|httpx)(==|>=|$)' /app/requirements.txt > /app/requirements.runtime.txt; \
	pip install --no-cache-dir -r /app/requirements.runtime.txt; \
	rm -f /app/requirements.runtime.txt

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
