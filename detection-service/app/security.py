
"""Authentication for the TRI-NAYANTRA detection microservice."""

import os
import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


bearer_scheme = HTTPBearer(auto_error=False)


def verify_detection_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """Allow requests only from clients with the shared service token."""

    expected_token = os.getenv("DETECTION_SERVICE_TOKEN", "").strip()

    # Fail closed if the server has not been configured securely.
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection service authentication is not configured.",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supplied_token = credentials.credentials

    if not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid detection service token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
