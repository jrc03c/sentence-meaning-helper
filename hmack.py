import hmac
import hashlib


def hmack(text, n=None):
    out = (
        hmac.new(
            bytes(text, "utf-8"), msg=bytes(text, "utf-8"), digestmod=hashlib.sha256,
        )
        .hexdigest()
        .lower()
    )

    if n is not None:
        return out[:n]

    return out
