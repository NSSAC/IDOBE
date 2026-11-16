"""Password hashing, session helpers, and a small in-memory login rate limiter."""

import re
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User

USERNAME_RE = re.compile(r"^[a-z0-9_.-]{2,64}$")
USERNAME_RULE = "2-64 characters: lowercase letters, digits, '_', '.', '-'"
MIN_PASSWORD_LENGTH = 8

_hasher = PasswordHasher()
# Verified against when the username doesn't exist, so response time doesn't reveal valid usernames.
_DUMMY_HASH = _hasher.hash("not-a-real-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def authenticate(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username.strip().lower()).one_or_none()
    if user is None:
        verify_password(_DUMMY_HASH, password)
        return None
    if not user.is_active or not verify_password(user.password_hash, password):
        return None
    if _hasher.check_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        db.commit()
    return user


class LoginRateLimiter:
    """Counts failed logins per client IP and per username within a sliding window.

    In-memory, so limits are per process. Good enough for a single-instance deployment; move to the
    database or Redis if the app is ever scaled out to several instances.
    """

    def __init__(self, max_failures: int, window_seconds: int):
        self.max_failures = max_failures
        self.window = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _recent(self, key: str) -> deque[float]:
        q = self._failures[key]
        cutoff = time.monotonic() - self.window
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def blocked(self, ip: str, username: str) -> bool:
        return any(len(self._recent(k)) >= self.max_failures for k in (f"ip:{ip}", f"user:{username}"))

    def record_failure(self, ip: str, username: str) -> None:
        now = time.monotonic()
        for k in (f"ip:{ip}", f"user:{username}"):
            self._recent(k).append(now)

    def reset(self, username: str) -> None:
        self._failures.pop(f"user:{username}", None)


login_limiter = LoginRateLimiter(settings.login_max_failures, settings.login_window_seconds)


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")
    user = db.get(User, user_id) if user_id is not None else None
    if user is None or not user.is_active:
        request.session.clear()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user
