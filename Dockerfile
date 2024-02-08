FROM python:3.10.13-slim-bullseye

#Install Cron
RUN apt-get update
RUN apt-get -y install cron

# Install python packages
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Add crontab file in the cron directory
ADD crontab /etc/cron.d/clean-cache-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/clean-cache-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Copy clear_cache file
COPY clear_cache.py /clear_cache.py

# COPY app directory ad go there
COPY som-app /som-app
WORKDIR /som-app

# Set variable indicating that we are in docker environemnt
ENV APP_ENVIRON=docker_env

# Run the command on container startup
#CMD cron && tail -f /var/log/cron.log; gunicorn -b 0.0.0.0:8050 app:server
CMD gunicorn -b 0.0.0.0:8050 app:server