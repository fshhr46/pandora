import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase
from pandora.data.mask_utils import mask_data


@dataclass
class EmailAddress(object):
    email: str
    email_masked: str


def mask_email(email: str):
    splitted_email = email.split("@")
    name = list(splitted_email[0])

    name = mask_data(name, range(3, 6))
    return f'{name}@{splitted_email[1]}'


class EmailAddressGenerator(DataGeneratorBase):
    def __init__(self, masking_func=None, locales=...) -> None:
        super().__init__(masking_func, locales=["zh_CN", "en_US"])

    def _generate(self):
        email = self.faker.ascii_email()
        return EmailAddress(
            email=email,
            email_masked=mask_email(email),
        )
