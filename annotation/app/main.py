from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from . import __version__
from .config import ANNOTATION_DIR, settings
from .routes import admin, annotations, auth, outbreaks, segments

STATIC_DIR = ANNOTATION_DIR / "app" / "static"

app = FastAPI(title="IDOBE expert annotation", version=__version__)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    session_cookie="idobe_session",
    max_age=settings.session_max_age,
    same_site="lax",
    https_only=settings.cookie_secure,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


for r in (auth.router, outbreaks.router, annotations.router, admin.router, segments.router):
    app.include_router(r)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/guide", StaticFiles(directory=ANNOTATION_DIR / "docs", html=True), name="guide")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True, "version": __version__}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/segments", include_in_schema=False)
def segments_page():
    return FileResponse(STATIC_DIR / "segments.html")


@app.get("/admin", include_in_schema=False)
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")
