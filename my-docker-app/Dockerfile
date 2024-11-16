FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python", "dataprocess.py && python model.py && python app.py"]
EXPOSE 5000

CMD ["python", "app.py"]