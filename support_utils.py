import smtplib
import sys
import time


def send_email(msg_body: str) -> None:
    """
    Sends the email to me!
    """
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("kafkabot9000@gmail.com", "thisisnotimportant")
    msg_body += "\n"
    msg_body += time.asctime(time.localtime(time.time()))
    msg_body += "\n"
    msg_body += sys.platform
    server.sendmail("kafkabot9000@gmail.com", "baytemiz@ucsc.edu", msg_body)
    server.quit()


def save_hyperparameters(filenames: list, path_to_file: str, breadcrumb="# BREADCRUMBS") -> None:
    """
    Saves the lines in between breadcrumbs in all the given filenames. This is used for saving hyperparameters for RL training.

    Parameters
    ----------
    breadcrumb: Writes the lines between {breadcrumb}_START and {breadcrumb}_END.

    """
    with open(path_to_file, "a") as dest:

        for filename in filenames:
            with open(filename, "r") as source:
                saving = False
                for line in source:
                    if line.strip() == f"{breadcrumb}_START":
                        dest.write("\n")
                        saving = True
                        continue
                    if line.strip() == f"{breadcrumb}_END":
                        saving = False
                        continue
                    if saving:
                        dest.write(line)
            print(f"{filename} hyperparameters have been saved!")
        print("Information saving is complete!")
