# Using lightweight alpine image
FROM python:3.9-slim

# Installing packages
RUN apt update
RUN pip install --no-cache-dir pipenv

# Defining working directory and adding source code
WORKDIR /usr/src/app
COPY Pipfile Pipfile.lock scripts/bootstrap.sh ./
COPY server ./server

# Install API dependencies
RUN pipenv install --system --deploy

# Start app
EXPOSE 5000
ENTRYPOINT ["/usr/src/app/bootstrap.sh"]
