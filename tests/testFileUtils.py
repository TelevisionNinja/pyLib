from src import fileUtils
import unittest
import os.path
import tarfile
import os


class FileUtilsTest(unittest.TestCase):
    def test_tar_utils(self):
        with tarfile.TarFile("test.tar", "w") as tar:
            tar.add("./tests/tarTest.txt")

        with tarfile.TarFile("test.tar") as tar:
            fileUtils.extractTar(tar, "./")

        extractedFileText = ""
        with open(os.path.abspath("./tarTest.txt")) as extractedFile:
            extractedFileText = extractedFile.read()
        self.assertEqual(extractedFileText, "test")

        os.remove("test.tar")
        os.remove("./tarTest.txt")

        with tarfile.TarFile("test.tar", "w") as tar:
            tar.add("./tests/tarTest.txt")
            os.makedirs("./tests/tarDirectory/")
            tar.add("./tests/tarDirectory/")

        with tarfile.TarFile("test.tar") as tar:
            fileUtils.extractTar(tar, "./")

        extractedFileText = ""
        with open(os.path.abspath("./tarTest.txt")) as extractedFile:
            extractedFileText = extractedFile.read()
        self.assertEqual(extractedFileText, "test")
        self.assertTrue(os.path.exists("./tarDirectory/"))

        #-----------------------------------------

        os.remove("test.tar")
        os.remove("./tarTest.txt")
        os.rmdir("./tarDirectory/")

        fileUtils.createTar("./tests", "test.tar", "./")
        self.assertTrue(os.path.exists("./test.tar"))

        with tarfile.TarFile("./test.tar") as tar:
            tar.extractall(path="./")

        extractedFileText = ""
        with open(os.path.abspath("./tarTest.txt")) as extractedFile:
            extractedFileText = extractedFile.read()
        self.assertEqual(extractedFileText, "test")
        self.assertTrue(os.path.exists("./tarDirectory/"))


if __name__ == '__main__':
    unittest.main()
