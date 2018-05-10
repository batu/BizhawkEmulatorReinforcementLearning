
def save_code(filenames: list, path_to_file: str, breadcrumb="# BREADCRUMBS") -> None:
    """
    Saves the lines in between breadcrumbs in all the given filenames. This is used for saving hyperparameters for RL training.

    Parameters
    ----------
    breadcrumb: Writes the lines between {breadcrumb}_START and {breadcrumb}_END.

    """
    print("started")

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
