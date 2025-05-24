from src import fileUtils
import unittest
import os.path
import tarfile
import zipfile
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

    def test_zip_utils(self):
        with zipfile.ZipFile("test.zip", "w") as zip:
            zip.write("./tests/zipTest.txt")

        with zipfile.ZipFile("test.zip") as zip:
            fileUtils.extractZip(zip, "./")

        extractedFileText = ""
        with open(os.path.abspath("./zipTest.txt")) as extractedFile:
            extractedFileText = extractedFile.read()
        self.assertEqual(extractedFileText, "zip test")

        os.remove("test.zip")
        os.remove("./zipTest.txt")

        with zipfile.ZipFile("test.zip", "w") as zip:
            zip.write("./tests/zipTest.txt")
            os.makedirs("./tests/zipDirectory/")
            zip.write("./tests/zipDirectory/")

        with zipfile.ZipFile("test.zip") as zip:
            fileUtils.extractZip(zip, "./")

        extractedFileText = ""
        with open(os.path.abspath("./zipTest.txt")) as extractedFile:
            extractedFileText = extractedFile.read()
        self.assertEqual(extractedFileText, "zip test")
        self.assertTrue(os.path.exists("./zipDirectory/"))

        #-----------------------------------------

        os.remove("test.zip")
        os.remove("./zipTest.txt")
        os.rmdir("./zipDirectory/")

        fileUtils.createZip("./tests", "test.zip", "./")
        self.assertTrue(os.path.exists("./test.zip"))

        with zipfile.ZipFile("./test.zip") as zip:
            zip.extractall(path="./")

        extractedFileText = ""
        with open(os.path.abspath("./zipTest.txt")) as extractedFile:
            extractedFileText = extractedFile.read()
        self.assertEqual(extractedFileText, "zip test")
        self.assertTrue(os.path.exists("./zipDirectory/"))


if __name__ == '__main__':
    unittest.main()
