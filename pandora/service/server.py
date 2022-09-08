from pathlib import Path
import os
import logging

from pandora.tools.common import init_logger


class Server(object):
    def __init__(self, flaskApp) -> None:
        self.flaskApp = flaskApp

    def run(self, args) -> None:
        # dirs
        self.output_dir = args.output_dir

        # default data dir log will be in $HOME/workspace/resource/datasets.
        if args.data_dir:
            self.test_data_dir = args.data_dir
        else:
            home = str(Path.home())
            self.test_data_dir = os.path.join(
                home, "workspace/resource/datasets")

        # default service log will be in the output_dir.
        if args.cache_dir:
            self.cache_dir = args.cache_dir
        else:
            home = str(Path.home())
            self.cache_dir = os.path.join(home, ".cache/torch/transformers")

        # default service log will be in the output_dir.
        if args.log_dir:
            self.log_path = os.path.join(args.log_dir, "service_log.txt")
        else:
            self.log_path = os.path.join(self.output_dir, "service_log.txt")

        # logs
        if args.log_level:
            log_level = logging.getLevelName(args.log_level)
        init_logger(log_file=self.log_path, log_file_level=log_level)

        # run
        self.flaskApp.run(host=args.host, port=args.port)
