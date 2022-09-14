from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase
from pandora.data.mask_utils import mask_data


@dataclass
class FinanceInfo(object):
    issuer: str
    card_num: str
    card_num_formatted: str
    card_num_masked: str
    card_expir_date: str
    card_cvc: str


def mask_card_num(card_num: str):
    return mask_data(card_num, range(6, 12))


class FinanceInfoGenerator(DataGeneratorBase):
    def __init__(self, masking_func=None, locales=...) -> None:
        super().__init__(masking_func, locales=["zh_CN", "en_US"])

    def _generate(self):
        info = self.faker.credit_card_full()
        issuer, name, card_num_expir_date, cvc, _ = info.split("\n")
        card_num, expir_date = card_num_expir_date.split(" ")
        card_num_formatted = f"{card_num[:4]} {card_num[4:8]} {card_num[8:12]} {card_num[12:]}"
        return FinanceInfo(
            issuer=issuer,
            card_num=card_num,
            card_num_formatted=card_num_formatted,
            card_num_masked=mask_card_num(card_num),
            card_expir_date=expir_date,
            card_cvc=cvc
        )
