FROM python:3.12-slim
WORKDIR /app
COPY testbed/controller/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY envs/ /app/envs/
COPY data/aws_calibration.csv /app/data/aws_calibration.csv
COPY testbed/controller/policy_controller.py /app/policy_controller.py
ENV CALIBRATION_CSV=/app/data/aws_calibration.csv
EXPOSE 8081
CMD ["uvicorn", "policy_controller:app", "--host", "0.0.0.0", "--port", "8081"]
