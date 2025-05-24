import tarfile
import os
import zipfile


def extractTar(tar: tarfile.TarFile, path: str):
    """
    extract TarFile relative to a path. only use this function if the tar was not created properly (using absolute paths instead of relative paths). use .extractall() if it was created properly

    ex:
    tar = ["/home/files/file1.txt", "/home/files/file2.txt"]
    using extractTar(tar, "/extracted") = ["/extracted/file1.txt", "/extracted/file2.txt"]
    using tar.extractall("/extracted") = ["/extracted/home/files/file1.txt", "/extracted/home/files/file2.txt"]
    """

    names = tar.getnames()
    length = len(names)

    if length == 0:
        return

    path = os.path.abspath(path)
    common_path = ""

    if length == 1: # get rid of parent directory
        common_path = os.path.dirname(names[0])
    else: # get rid of shared parent directory
        common_path = os.path.commonpath(names)

    for member in tar.getmembers():
        directory_split = member.name.split(common_path, maxsplit=1)

        if len(directory_split) > 1: # could be a file
            member.name = directory_split[1].lstrip(os.path.sep)

        tar.extract(member, path)


def createTar(dir_to_archive: str, archive_name: str, archive_path: str):
    """
    dir_to_archive: the directory of files and directories you want to be put in a tar
    archive_name: the name of the tar (include the extension)
    archive_path: the location of where you want to create the tar
    """
    with tarfile.TarFile(os.path.join(archive_path, archive_name), "w") as tar:
        fullpath = os.path.abspath(dir_to_archive)

        for file in os.listdir(fullpath):
            fullFilePath = os.path.join(fullpath, file)
            # arcname is the relative path
            tar.add(fullFilePath, arcname=file)


def extractZip(zip: zipfile.ZipFile, path: str, password: bytes = None):
    """
    extract TarFile relative to a path. only use this function if the zip was not created properly (using absolute paths instead of relative paths). use .extractall() if it was created properly

    ex:
    zip = ["/home/files/file1.txt", "/home/files/file2.txt"]
    using extractZip(zip, "/extracted") = ["/extracted/file1.txt", "/extracted/file2.txt"]
    using zip.extractall("/extracted") = ["/extracted/home/files/file1.txt", "/extracted/home/files/file2.txt"]
    """

    names = zip.namelist()
    length = len(names)

    if length == 0:
        return

    path = os.path.abspath(path)
    common_path = ""

    if length == 1: # get rid of parent directory
        common_path = os.path.dirname(names[0])
    else: # get rid of shared parent directory
        common_path = os.path.commonpath(names)

    for member in zip.infolist():
        directory_split = member.filename.split(common_path, maxsplit=1)

        if len(directory_split) > 1: # could be a file
            member.filename = directory_split[1].lstrip(os.path.sep)

        zip.extract(member, path, pwd=password)


def createZip(dir_to_archive: str, archive_name: str, archive_path: str):
    """
    dir_to_archive: the directory of files and directories you want to be put in a zip
    archive_name: the name of the zip (include the extension)
    archive_path: the location of where you want to create the zip

    python can't create encrypted zip files
    """
    with zipfile.ZipFile(os.path.join(archive_path, archive_name), "w") as zip:
        fullpath = os.path.abspath(dir_to_archive)

        for file in os.listdir(fullpath):
            fullFilePath = os.path.join(fullpath, file)
            # arcname is the relative path
            zip.write(fullFilePath, arcname=file)
