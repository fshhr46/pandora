

from pathlib import Path
import os
import json

from pandora.dataset.sentence_data import Dataset
from pandora.tools.common import init_logger
import pandora.tools.report as report


def main():
    resource_dir = os.path.join(Path.home(), "workspace", "resource")
    datasets = [
        Dataset.column_data,
        # Dataset.short_sentence,
        # Dataset.long_sentence,
    ]
    datasets.sort()
    dataset_names = "_".join(datasets)
    output_dir = os.path.join(resource_dir,
                              f"outputs/bert-base-chinese/{dataset_names}/")
    labdels_path = os.path.join(output_dir, "index_to_name.json")
    label_list = list(json.load(open(labdels_path)).values())

    predict_dir = os.path.join(output_dir, "predict")
    predict_path = os.path.join(predict_dir, "test_submit.json")
    # truth_path = os.path.join(resource_dir, "CLUEdatasets/cluener/dev.json")
    truth_path = predict_path

    pre_lines = [json.loads(line.strip())
                 for line in open(predict_path) if line.strip()]
    truth_lines = [json.loads(line.strip())
                   for line in open(truth_path) if line.strip()]
    # validation
    assert len(pre_lines) == len(truth_lines)

    report.build_report(pre_lines=pre_lines,
                        truth_lines=truth_lines,
                        label_list=label_list,
                        report_dir=predict_dir)

    report.build_report_sklearn(pre_lines=pre_lines,
                                truth_lines=truth_lines,
                                label_list=label_list,
                                report_dir=None)


if __name__ == "__main__":
    init_logger()
    main()
