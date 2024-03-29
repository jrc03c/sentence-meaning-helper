from unittest import TestCase

from .sha256 import sha256


class SHA256TestCase(TestCase):
    def test(self):
        a = sha256("Hello, world!")
        b = sha256("Hello, world!")
        c = sha256("Goodbye, world!")
        self.assertTrue(a == b)
        self.assertFalse(a == c)
        self.assertTrue(len(a) == len(c))

        for v in [a, b, c]:
            self.assertTrue(type(v) == str)

        wrongs = [
            0,
            1,
            2.3,
            -2.3,
            True,
            False,
            None,
            [2, 3, 4],
            [
                [2, 3, 4],
                [5, 6, 7],
            ],
            lambda x: x,
            sha256,
            {"hello": "world"},
        ]

        for v in wrongs:
            failed = False

            try:
                sha256(v)

            except:
                failed = True

            self.assertTrue(failed)
