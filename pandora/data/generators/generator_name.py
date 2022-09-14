import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase
from pandora.data.resource.names import NAMES_CN, NAMES_US
from pandora.data.mask_utils import mask_data


@dataclass
class Name(object):
    name_cn: str
    name_cn_masked: str
    name_us: str


class NameGenerator(DataGeneratorBase):

    def _generate(self):
        name_cn = random.choice(NAMES_CN)
        return Name(
            name_cn=name_cn,
            name_cn_masked=mask_data(name_cn, [1]),
            name_us=random.choice(NAMES_US),
        )
