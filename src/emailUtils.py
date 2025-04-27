import email
import email.message
import os.path


def extractAttachments(emailFile: str, extractionDirectory: str):
    with open(os.path.abspath(emailFile)) as emailFileObject:
        message = email.message_from_file(emailFileObject)

        for part in message.walk():
            fileName = part.get_filename()
            contentDisposition = part.get_content_disposition()

            if fileName is not None and contentDisposition is not None and contentDisposition.startswith("attachment"):
                with open(os.path.join(os.path.abspath(extractionDirectory), fileName), "wb") as attachment:
                    attachment.write(part.get_payload(decode=True))


def writeEmail(filePath: str, receivers: list[str], sender: str, subject: str, attachmentPaths: list[str]):
    message = email.message.EmailMessage()
    message["To"] = ", ".join(receivers)
    message["From"] = sender
    message["Subject"] = subject

    for attachment in attachmentPaths:
        with open(os.path.abspath(attachment), "rb") as attachmentFile:
            fileName = os.path.basename(attachment)
            extension = fileName.split(".")[-1]
            message.add_attachment(attachmentFile.read(), maintype = "application", subtype = extension, filename = fileName)

    with open(os.path.abspath(filePath), "xt") as emailFile:
        emailFile.write(message.as_string())
