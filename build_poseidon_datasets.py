import json

import pathlib
import os
import pandora.dataset.poseidon_data as poseidon_data
from pandora.tools.common import init_logger, logger

if __name__ == "__main__":
    init_logger()
    dataset_name = "meta_data"
    dataset_name = "mixed_data"
    num_folds = 3

    # create output dir
    output_dir_name = dataset_name
    # if num_folds > 0:
    #     output_dir_name = f"{output_dir_name}_{num_folds}"
    output_dir = os.path.join(
        pathlib.Path.home(), "workspace", "resource", "datasets", output_dir_name)

    if os.path.exists(output_dir):
        raise Exception(f"output_dir {output_dir} already exists")
    result = poseidon_data.partition_poseidon_dataset(
        dataset_path=f"./test_data/{dataset_name}.json",
        output_dir=output_dir,
        min_samples=1,
        data_ratios={"train": 0.6, "dev": 0.2, "test": 0.2},
        num_folds=num_folds,
        seed=42,
    )

    logger.info(json.dumps(result, indent=4, ensure_ascii=False,
                           cls=poseidon_data.DatasetJSONEncoder))
