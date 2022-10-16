import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase


@dataclass
class AddressInfo(object):
    home_add: str
    jiating_dizhi: str
    home_postal_code: str
    work_add: str
    gongzuo_dizhi: str
    work_postal_code: str



class AddressInfoGenerator(DataGeneratorBase):
    def _generate(self):
        return AddressInfo(
            home_add=self.faker.address(),
            jiating_dizhi=self.faker.address(),
            home_postal_code=self.faker.postcode(),
            work_add=self.faker.address(),
            gongzuo_dizhi=self.faker.address(),
            work_postal_code=self.faker.postcode(),
        )
