FROM python:3.12-alpine
WORKDIR /app
COPY tokens.py server.py ./
ENV INVITE_BIND=0.0.0.0
ENV INVITE_PORT=8090
EXPOSE 8090
CMD ["python", "server.py"]
