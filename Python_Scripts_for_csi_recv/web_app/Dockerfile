FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CSI_DB_PATH=/data/app.sqlite3 \
    CSI_MODEL_DIR=/models/har \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    CSI_COOKIE_SECURE=true

WORKDIR /app
COPY web_app/requirements.lock.txt /app/web_app/requirements.lock.txt
RUN pip install --no-cache-dir -r /app/web_app/requirements.lock.txt \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown app:app /data

COPY . /app
USER app
EXPOSE 8000

CMD ["uvicorn", "web_app.server:app", "--host", "0.0.0.0", "--port", "8000"]
