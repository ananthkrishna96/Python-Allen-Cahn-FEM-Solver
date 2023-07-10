FROM python:3.9

WORKDIR /app

RUN pip install numpy sympy scipy matplotlib

COPY FEMAC2D.py /
COPY test_FEMAC2D.py /
COPY . .

CMD ["python", "test_FEMAC2D.py"]
