FROM python:3.10-slim

RUN pip install mlflow==2.12.1

EXPOSE 5000

CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/ubuntu/kpps/mlops/mlopszoomcamp2024/mlops_zoomcamp_hw3/mlflow/mlflow.db", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]
