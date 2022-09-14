import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase
from pandora.data.resource.majors import MAJORS
from pandora.data.resource.universities import UNIVERSITIES


@dataclass
class EduInfo(object):
    university: str
    major: str


class EduInfoGenerator(DataGeneratorBase):

    def _generate(self):
        return EduInfo(
            random.choice(list(UNIVERSITIES.values())),
            random.choice(list(MAJORS.values())),
        )
