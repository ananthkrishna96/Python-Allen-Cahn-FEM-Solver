FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy sympy scipy matplotlib \
    && rm -rf /var/lib/apt/lists/*

COPY FEMAC2D.py test_FEMAC2D.py /app/

CMD ["python", "test_FEMAC2D.py"]
