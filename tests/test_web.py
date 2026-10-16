import unittest

from mm_jailbreak_guard.web import _looks_like_supported_image


class WebValidationTests(unittest.TestCase):
    def test_image_signatures(self):
        self.assertTrue(_looks_like_supported_image(b"\x89PNG\r\n\x1a\nrest", ".png"))
        self.assertTrue(_looks_like_supported_image(b"\xff\xd8\xffrest", ".jpg"))
        self.assertTrue(_looks_like_supported_image(b"RIFF1234WEBPrest", ".webp"))
        self.assertTrue(_looks_like_supported_image(b"BMrest", ".bmp"))

    def test_disguised_text_is_rejected(self):
        self.assertFalse(_looks_like_supported_image(b"not an image", ".png"))


if __name__ == "__main__":
    unittest.main()
