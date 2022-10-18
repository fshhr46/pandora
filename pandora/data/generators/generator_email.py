import random
from dataclasses import dataclass
from turtle import position

from pandora.data.generators.generator_base import DataGeneratorBase
import pandora.data.mask_utils as mask_utils


@dataclass
class EmailAddress(object):
    email_add: str
    email_add_masked: str
    col_1: str


def mask_email(email: str, positions=range(3, 6), mask_char="*"):
    splitted_email = email.split("@")
    name = list(splitted_email[0])

    name = mask_utils.mask_data(name, positions=positions, mask_char=mask_char)
    return f'{name}@{splitted_email[1]}'


class EmailAddressGenerator(DataGeneratorBase):
    def __init__(self, masking_func=None, locales=..., *args, **kwargs) -> None:
        super().__init__(*args, locales=["zh_CN", "en_US"], **kwargs)

    def _generate(self):
        email = self.faker.ascii_email()
        return EmailAddress(
            email_add=email,
            email_add_masked=mask_email(email),
            col_1=email,
        )

    def _generate_test(self):
        email = self.faker.ascii_email()

        mask_char = random.choice(mask_utils.get_mask_chars())

        start = random.randint(0, 3)
        random_positions = range(start, start + 2)
        return EmailAddress(
            email_add=email,
            email_add_masked=mask_email(
                email, positions=random_positions, mask_char=mask_char),
            col_1=email,
        )
