from src import emailUtils
import unittest
import os.path


class EmailUtilsTest(unittest.TestCase):
    def test_extractAttachments(self):
        emailUtils.writeEmail("./email.txt", ["1@test.test", "2@test.test"], "3@test.test", "test", ["./tests/attachmentTest.txt"])
        emailUtils.extractAttachments("./email.txt", "./")

        attachmentText = ""
        with open(os.path.abspath("./attachmentTest.txt")) as attachmentFile:
            attachmentText = attachmentFile.read()
        self.assertEqual(attachmentText, "test")


if __name__ == '__main__':
    unittest.main()
