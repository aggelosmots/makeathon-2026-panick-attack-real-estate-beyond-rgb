FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    default-jdk-headless \
    libexpat1 \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
RUN mkdir -p build/classes \
    && javac -d build/classes $(find src/main/java -name "*.java")

ENV PYTHONPATH=/app

# The compose file chooses the final command.
CMD ["java", "-cp", "/app/build/classes", "com.beyondrgb.AppServer"]
