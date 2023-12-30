import io
import os
import pandas as pd
import subprocess


class KaggleCompetitionDownloader:
    def __init__(self, competition_name: str) -> None:
        self.competition_name = competition_name

    def list_kernels(self):
        current_page = 1
        finished = False
        output_frames = []
        while not finished:
            print(f"Downloading page {current_page}")
            res = subprocess.run(
                [
                    "kaggle",
                    "kernels",
                    "list",
                    "-v",
                    "--competition",
                    self.competition_name,
                    "-p",
                    str(current_page),
                    "--page-size",
                    "100",
                ],
                capture_output=True,
            )
            output = res.stdout.decode("utf-8")
            if "Not found" in output:
                finished = True
            else:
                output_frames.append(pd.read_csv(io.StringIO(output)))
                current_page += 1
        if len(output_frames) == 0:
            raise ValueError("No kernels found")
        return pd.concat(output_frames)

    def download_kernel(self, kernels: pd.DataFrame, path: str):
        for _, row in kernels.iterrows():
            subprocess.run(["kaggle", "kernels", "pull", row["ref"]], cwd=str(path))

    def download_all_kernels(self, path: str):
        kernels = self.list_kernels()
        self.download_kernel(kernels, path)

    def convert_all_kernels(self, input_path: str, output_path: str):
        kernels = os.listdir(input_path)
        for kernel in kernels:
            print(f"Converting {kernel}")
            notebook_path = os.path.join(input_path, kernel)
            res = subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--output-dir",
                    output_path,
                    notebook_path,
                ],
                capture_output=True,
            )
            if res.returncode != 0:
                print(f"Failed to convert {kernel}")
                print(res.stderr.decode("utf-8"))
