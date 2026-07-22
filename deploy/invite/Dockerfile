FROM python:3.12-alpine
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY tokens.py mail.py rate_limit.py server.py ./
COPY templates/ ./templates/
ENV INVITE_BIND=0.0.0.0
ENV INVITE_PORT=8090
EXPOSE 8090
CMD ["python", "server.py"]
