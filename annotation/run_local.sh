#!/bin/sh
# Start the annotation app on this computer, from any terminal:
#
#   ./annotation/run_local.sh            # http://localhost:8000, opens your browser
#   ./annotation/run_local.sh --lan      # also reachable from other devices on your network
#   PORT=8080 ./annotation/run_local.sh
#
# First run sets everything up: Python environment, .env with a secret key, database, outbreak data,
# and an admin account (its password is printed once). Stop the server with Ctrl+C.
set -e
cd "$(dirname "$0")"

HOST=127.0.0.1
[ "$1" = "--lan" ] && HOST=0.0.0.0
PORT="${PORT:-8000}"

if [ ! -x .venv/bin/python ]; then
  echo "Setting up the Python environment (first run only)..."
  if command -v uv >/dev/null 2>&1; then
    uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
  else
    PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
    "$PY" -m venv .venv && .venv/bin/pip install -q -r requirements.txt
  fi
fi

if [ ! -f .env ]; then
  cp .env.example .env
  KEY=$(.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(32))")
  sed -i.bak "s|^SECRET_KEY=\$|SECRET_KEY=$KEY|" .env && rm -f .env.bak
  echo "Created annotation/.env"
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Is the app already running? Otherwise try: PORT=8080 $0"
  exit 1
fi

.venv/bin/alembic upgrade head 2>&1 | grep -v "^INFO" || true
COUNTS=$(.venv/bin/python -c "
from sqlalchemy import func, select
from app.db import SessionLocal
from app.models import Outbreak, SegmentationRun, User
with SessionLocal() as db:
    print(db.scalar(select(func.count()).select_from(Outbreak)), db.scalar(select(func.count()).select_from(User)),
          db.scalar(select(func.count()).select_from(SegmentationRun)))")
set -- $COUNTS
[ "$1" = "0" ] && .venv/bin/python scripts/load_outbreaks.py
# Segment review starts with the published (epidemicKabu) windows of the respiratory series.
[ "$3" = "0" ] && .venv/bin/python scripts/load_segments.py --legacy
if [ "$2" = "0" ]; then
  echo "Creating an admin account. Note the password, it is shown only once:"
  .venv/bin/python scripts/create_user.py admin --admin --generate
fi

URL="http://localhost:$PORT"
echo ""
echo "Expert annotation app: $URL   (segment review: $URL/segments, admin page: $URL/admin)"
if [ "$HOST" = "0.0.0.0" ]; then
  IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | cut -d' ' -f1)
  [ -n "$IP" ] && echo "On other devices on this network: http://$IP:$PORT"
fi
echo "Add experts with:  annotation/.venv/bin/python annotation/scripts/create_user.py <name> --generate"
echo "Press Ctrl+C to stop."
echo ""

if [ -z "$NO_BROWSER" ]; then
  ( for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      if curl -sf "$URL/healthz" >/dev/null 2>&1; then
        (command -v open >/dev/null && open "$URL") || (command -v xdg-open >/dev/null && xdg-open "$URL") || true
        break
      fi
      sleep 0.5
    done ) &
fi

exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT"
