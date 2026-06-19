import time
import requests

class CognitoAuth:
    """
    Gestiona el token M2M (Client Credentials) de Cognito.
    Cachea el token y lo renueva automáticamente antes de que venza.

    Uso:
        auth = CognitoAuth(
            token_url  = "https://g2-auth-prod.auth.us-east-2.amazoncognito.com/oauth2/token",
            client_id  = "TU_CLIENT_ID",
            client_secret = "TU_CLIENT_SECRET",
            scope      = "https://g2-api/read https://g2-api/write",
        )

        # Antes de cada request:
        headers = auth.auth_header()
        response = requests.get("https://<api-url>/events", headers=headers)
    """

    _EXPIRY_BUFFER = 60  # renueva el token 60 segundos antes de que venza

    def __init__(self, token_url: str, client_id: str, client_secret: str, scope: str):
        self._token_url    = token_url
        self._client_id    = client_id
        self._client_secret = client_secret
        self._scope        = scope
        self._access_token: str | None = None
        self._expires_at: float = 0

    def _fetch_token(self) -> None:
        response = requests.post(
            self._token_url,
            data={
                "grant_type":    "client_credentials",
                "client_id":     self._client_id,
                "client_secret": self._client_secret,
                "scope":         self._scope,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
        self._access_token = body["access_token"]
        self._expires_at   = time.time() + body["expires_in"] - self._EXPIRY_BUFFER

    def token(self) -> str:
        if self._access_token is None or time.time() >= self._expires_at:
            self._fetch_token()
        return self._access_token

    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.token()}"}
