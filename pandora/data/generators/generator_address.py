import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase


@dataclass
class AddressInfo(object):
    living_address: str


class AddressInfoGenerator(DataGeneratorBase):
    def _generate(self):
        return AddressInfo(
            living_address=self.faker.address()
        )
