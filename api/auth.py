"""
api/auth.py
-----------
JWT authentication for the CARIS API.

WHY JWT:
  Every API endpoint that returns sensitive plant data
  must be authenticated. JWT (JSON Web Token) is the
  industry standard for stateless API auth.

  Flow:
    1. Engineer calls POST /api/login with username/password
    2. API returns a JWT token (valid 8 hours = one shift)
    3. Engineer includes token in every request header:
       Authorization: Bearer <token>
    4. FastAPI validates token on every protected route

  In production at CPChem, this would integrate with
  Azure Active Directory (SSO). Same JWT pattern, different
  identity provider.

DEMO CREDENTIALS (for interview demo only):
  username: engineer | password: cpchem2025
  username: admin    | password: caris2025
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET", "caris-cpchem-secret-key-change-in-prod")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 8  # one shift

# demo users (in production: Azure AD or LDAP)
DEMO_USERS = {
    "engineer": {
        "password": "cpchem2025",
        "role": "reliability_engineer",
        "plant": "cedar_bayou",
    },
    "admin": {
        "password": "caris2025",
        "role": "admin",
        "plant": "cedar_bayou",
    },
}

security = HTTPBearer()


def create_token(username: str, role: str) -> str:
    """Create a JWT token for authenticated user."""
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "sub":  username,
        "role": role,
        "exp":  expire,
        "iat":  datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Dependency injected into every protected FastAPI route.
    Returns the decoded token payload if valid.
    Raises 401 if invalid or expired.
    """
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalid or expired"
        )


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Verify username/password and return user dict if valid."""
    user = DEMO_USERS.get(username)
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None