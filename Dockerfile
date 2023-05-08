FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "peptide_forest_3", "-c", "config.json", "-o", "output.csv"]