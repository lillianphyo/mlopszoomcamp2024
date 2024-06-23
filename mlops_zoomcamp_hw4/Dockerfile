FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy the script into the container
COPY script.py .

# Install dependencies
RUN pip install pandas pyarrow scikit-learn

# Command to run the script with arguments for May 2023
CMD ["python", "script.py", "--year", "2023", "--month", "5"]

