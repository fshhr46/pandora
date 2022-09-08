import os
from pathlib import Path
import pandora.dataset.sentence_data as sentence_data
from pandora.tools.common import init_logger


if __name__ == "__main__":
    init_logger()
    resource_dir = os.path.join(Path.home(), "workspace", "resource")
    seed = 42
    sentence_data.build_datasets(resource_dir, seed=seed)
    sentence_data.split_test_set_from_train(
        resource_dir,
        data_ratios={"train": 0.6, "dev": 0.2, "test": 0.2},
        seed=seed)
