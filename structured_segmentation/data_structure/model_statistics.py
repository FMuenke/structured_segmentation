"""
This module contains the ModelStatistics Class which is used to
evaluate models on multiple images and stores intermediate results.
"""


class ModelStatistics:
    """
    This Class handles and stores results produced by a segmentation model
    """
    def __init__(self, color_coding):
        """
        Initialize the Object with information on the classes to check
        """
        self.raw_statistics = {"total": {"tp": 0, "fp": 0, "fn": 0}}
        self.result = {}
        self.color_coding = color_coding

        for cls in color_coding:
            self.raw_statistics[cls] = {
                "tp": 0,
                "fn": 0,
                "fp": 0,
            }

    def __str__(self):
        """
        Built-In function to generate a report as string with all available information
        """
        return self.generate_r_string()

    def count(self, cls, error_type, n_samples=1):
        """
        This function counts the produced results and stores them correctly
        cls: Name of class the is registered
        error_type: type [tp, fp, fn]
        n_samples: amount of samples counted
        """
        self.raw_statistics[cls][error_type] += n_samples
        self.raw_statistics["total"][error_type] += n_samples

    def eval(self):
        """
        This function computes the final results from all intermediate results
        """
        for cls in self.raw_statistics:
            rec = self.raw_statistics[cls]["tp"] / (
                    self.raw_statistics[cls]["tp"] + self.raw_statistics[cls]["fn"] + 1e-5)
            pre = self.raw_statistics[cls]["tp"] / (
                    self.raw_statistics[cls]["tp"] + self.raw_statistics[cls]["fp"] + 1e-5)
            f_1 = 2 * (pre * rec) / (pre + rec + 1e-5)
            iou = self.raw_statistics[cls]["tp"] / (
                    self.raw_statistics[cls]["tp"] + self.raw_statistics[cls]["fn"] + self.raw_statistics[cls]["fp"])
            self.result[cls] = {
                "rec": rec,
                "pre": pre,
                "f_1": f_1,
                "iou": iou,
            }

    def generate_r_string(self):
        """
        Format the final results into a nice report
        """
        r_string = ""
        for cls in self.result:
            pre = self.result[cls]["pre"]
            rec = self.result[cls]["rec"]
            f_1 = self.result[cls]["f_1"]
            iou = self.result[cls]["iou"]
            r_string += "------------------\n"
            r_string += f"{cls}-PRE: {pre}\n"
            r_string += f"{cls}-REC: {rec}\n"
            r_string += f"{cls}-F_1: {f_1}\n"
            r_string += f"{cls}-JAC: {iou}\n"
            r_string += "------------------\n"
        return r_string

    def show(self):
        """
        Display the final report in the console
        """
        r_string = self.generate_r_string()
        print(r_string)

    def write_report(self, path_to_report):
        """
        Write the final report to a file
        """
        with open(path_to_report, "w") as report_file:
            report_file.write(self.generate_r_string())
