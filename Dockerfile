FROM python:3.11-slim

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
