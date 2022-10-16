import time
from dataclasses import dataclass
import random

from pandora.data.generators.generator_base import DataGeneratorBase
from pandora.data.mask_utils import mask_data


def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date():
    return str_time_prop("1960-01-01", "2000-12-30", "%Y-%m-%d", random.random())


@dataclass
class OperationInfo(object):
    acc_numb: str
    acc_open_date: str
    is_active_acc: int
    service_type: int


def mask_card_num(card_num: str):
    return mask_data(card_num, range(6, 12))


class OperationInfoGenerator(DataGeneratorBase):
    def __init__(self, masking_func=None, locales=..., *args, **kwargs) -> None:
        super().__init__(*args, locales=["zh_CN", "en_US"], **kwargs)

    def _generate(self):
        return OperationInfo(
            acc_numb=str(random.randint(100000, 999999)),
            acc_open_date=random_date(),
            is_active_acc=random.randint(0, 1),
            service_type=random.randint(0, 4),
        )
