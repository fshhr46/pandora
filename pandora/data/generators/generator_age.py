import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase


@dataclass
class Age(object):
    age1: int
    age2: int
    age3: int


class AgeGenerator(DataGeneratorBase):

    def _generate(self):
        age = random.randint(20, 100)
        return Age(
            age1=age,
            age2=age,
            age3=random.randint(20, 100),
        )
