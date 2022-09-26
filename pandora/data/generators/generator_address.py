import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase


@dataclass
class AddressInfo(object):
    residential_add: str
    jiating_dizhi: str
    work_add: str
    gongzuo_dizhi: str


class AddressInfoGenerator(DataGeneratorBase):
    def _generate(self):
        return AddressInfo(
            residential_add=self.faker.address(),
            jiating_dizhi=self.faker.address(),
            work_add=self.faker.address(),
            gongzuo_dizhi=self.faker.address(),
        )
