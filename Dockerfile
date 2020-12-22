FROM python:3.8-slim-buster
WORKDIR /usr/src/app
COPY requirements.txt .
RUN apt-get -y update
RUN apt-get -y install git
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN bash setup.sh
CMD ["streamlit", "run", "k_means_demo.py"]