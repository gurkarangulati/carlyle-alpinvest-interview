# The Cloud Run container: Python + CPU PyTorch + the model baked in.
FROM python:3.12-slim
WORKDIR /srv
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY model/ model/
ENV MODEL_DIR=/srv/model PORT=8080
# Run as a non-root user.
RUN useradd --create-home --uid 1000 app && chown -R app /srv
USER app
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
