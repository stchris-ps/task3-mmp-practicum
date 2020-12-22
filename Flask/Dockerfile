FROM python:3.8-slim

COPY useful/libs.txt /root/server/useful/libs.txt

RUN chown -R root:root /root/server/useful/libs.txt

WORKDIR /root/server/useful
RUN pip3 install -r libs.txt

COPY useful/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]
