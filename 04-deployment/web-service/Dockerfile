FROM python:3.9.7-slim

# update pip
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

# install dependencies to the system
# since we are already in an isolated environment, we do'n need to create an extra virtual env
RUN pipenv install --system --deploy 

COPY ["predict_flask.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_flask:app"]

