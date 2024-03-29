import hashlib
import hmac


def sha256(text):
    assert (
        type(text) == str
    ), "The value passed into the `sha256` function must be a string!"

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
