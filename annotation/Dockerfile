# Build from the repository root so the outbreak CSV is available to the loader:
#   docker build -f annotation/Dockerfile -t idobe-annotation .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    COOKIE_SECURE=true \
    # Trust X-Forwarded-* from the hosting platform's proxy (real client IPs for the login rate limit).
    FORWARDED_ALLOW_IPS="*"

WORKDIR /srv/annotation
COPY annotation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY annotation/ .
COPY raw_data/outbreaks_disease_location.csv /srv/raw_data/outbreaks_disease_location.csv
COPY pipeline/ /srv/pipeline/

RUN useradd --system --no-create-home app && chown -R app /srv/annotation
USER app

EXPOSE 8000
CMD ["sh", "start.sh"]
