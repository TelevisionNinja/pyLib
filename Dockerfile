FROM python:latest
WORKDIR /pyLib
COPY ./ ./
CMD ["python", "-m", "unittest"]
