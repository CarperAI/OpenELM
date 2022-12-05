# Using lightweight alpine image
FROM python:3.9-slim

# Installing packages
RUN apt-get update && \
    apt-get install -y git gcc g++ swig
RUN pip install --no-cache-dir pipenv

# Defining working directory and adding source code
WORKDIR /usr/src/app
COPY . /OpenELM/
COPY ./elm/sandbox/Pipfile ./elm/sandbox/Pipfile.lock ./elm/sandbox/scripts/bootstrap.sh ./
COPY ./elm/sandbox/server ./server


# Install API dependencies
RUN pipenv install --system --deploy

# Install local ELM package
RUN pipenv install /OpenELM/.


# Start app
EXPOSE 5000
ENTRYPOINT ["/usr/src/app/bootstrap.sh"]
