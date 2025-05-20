FROM python:3.10
RUN pip install -r requirements.txt
ADD ./Models ./Models
ADD server.py server.py
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
