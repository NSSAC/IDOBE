from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..analysis import latest_ids_subquery
from ..auth import authenticate, current_user, login_limiter
from ..config import settings
from ..db import get_db
from ..models import Annotation, User
from ..schemas import LoginIn

router = APIRouter(prefix="/api", tags=["auth"])


def _user_out(db: Session, user: User) -> dict:
    latest = latest_ids_subquery()
    counts = dict(
        db.execute(
            select(Annotation.label, func.count())
            .join(latest, Annotation.id == latest.c.id)
            .where(Annotation.user_id == user.id)
            .group_by(Annotation.label)
        ).all()
    )
    skipped = counts.pop("skip", 0)
    return {
        "username": user.username,
        "display_name": user.display_name or user.username,
        "is_admin": user.is_admin,
        "labeled": sum(counts.values()),
        "skipped": skipped,
        "target_annotations": settings.target_annotations,
    }


@router.post("/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "unknown"
    username = body.username.strip().lower()
    if login_limiter.blocked(ip, username):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed attempts; try again later")
    user = authenticate(db, username, body.password)
    if user is None:
        login_limiter.record_failure(ip, username)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    login_limiter.reset(username)
    request.session.clear()
    request.session["user_id"] = user.id
    return _user_out(db, user)


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _user_out(db, user)
