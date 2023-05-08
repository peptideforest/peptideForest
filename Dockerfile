FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .

CMD ["python", "-m", "peptide_forest_3", "-c", "./docker_test_data/config.json", "-o", "./docker_test_data/output.csv"]