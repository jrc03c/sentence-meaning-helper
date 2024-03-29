import hashlib
import hmac


def sha256(text):
    out = (
        hmac.new(
            bytes(text, "utf-8"),
            msg=bytes(text, "utf-8"),
            digestmod=hashlib.sha256,
        )
        .hexdigest()
        .lower()
    )

    return out
