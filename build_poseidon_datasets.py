import json
import pandora.dataset.poseidon_data as poseidon_data
from pandora.tools.common import init_logger, logger

if __name__ == "__main__":
    init_logger()
    result = poseidon_data.partition_poseidon_dataset(
        dataset_path="./test_data/dataset.json",
        output_dir="./test_data/dummy/",
        min_samples=203,
        data_ratios={"train": 0.6, "dev": 0.2, "test": 0.2},
        seed=42,
    )

    logger.info(json.dumps(result, indent=4, ensure_ascii=False,
                           cls=poseidon_data.DatasetJSONEncoder))
