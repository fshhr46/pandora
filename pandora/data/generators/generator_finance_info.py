from dataclasses import dataclass
import random

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
    def __init__(self, masking_func=None, locales=..., *args, **kwargs) -> None:
        super().__init__(*args, locales=["zh_CN", "en_US"], **kwargs)

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


@dataclass
class FinanceReport(object):
    q1_2022_sales: str
    q2_2022_sales: str
    q3_2022_sales: str
    q4_2022_sales: str
    total_2022_sales: str
    q1_2022_sales_growth: str
    q2_2022_sales_growth: str
    q3_2022_sales_growth: str
    q4_2022_sales_growth: str
    total_2022_sales_growth: str
    q1_2021_sales: str
    q2_2021_sales: str
    q3_2021_sales: str
    q4_2021_sales: str
    total_2021_sales: str
    q1_2021_sales_growth: str
    q2_2021_sales_growth: str
    q3_2021_sales_growth: str
    q4_2021_sales_growth: str
    total_2021_sales_growth: str
    q1_2022_rev: str
    q2_2022_rev: str
    q3_2022_rev: str
    q4_2022_rev: str
    total_2022_rev: str
    q1_2022_rev_growth: str
    q2_2022_rev_growth: str
    q3_2022_rev_growth: str
    q4_2022_rev_growth: str
    total_2022_rev_growth: str
    q1_2021_rev: str
    q2_2021_rev: str
    q3_2021_rev: str
    q4_2021_rev: str
    total_2021_rev: str
    q1_2021_rev_growth: str
    q2_2021_rev_growth: str
    q3_2021_rev_growth: str
    q4_2021_rev_growth: str
    total_2021_rev_growth: str


class FinanceReportGenerator(DataGeneratorBase):
    def _generate(self):
        return FinanceReport(
            q1_2022_sales=random.randint(1000000, 9999999),
            q2_2022_sales=random.randint(1000000, 9999999),
            q3_2022_sales=random.randint(1000000, 9999999),
            q4_2022_sales=random.randint(1000000, 9999999),
            total_2022_sales=random.randint(1000000, 9999999),
            q1_2022_sales_growth=random.randint(100000, 999999),
            q2_2022_sales_growth=random.randint(100000, 999999),
            q3_2022_sales_growth=random.randint(100000, 999999),
            q4_2022_sales_growth=random.randint(100000, 999999),
            total_2022_sales_growth=random.randint(1000000, 9999999),
            q1_2021_sales=random.randint(1000000, 9999999),
            q2_2021_sales=random.randint(1000000, 9999999),
            q3_2021_sales=random.randint(1000000, 9999999),
            q4_2021_sales=random.randint(1000000, 9999999),
            total_2021_sales=random.randint(1000000, 9999999),
            q1_2021_sales_growth=random.randint(100000, 999999),
            q2_2021_sales_growth=random.randint(100000, 999999),
            q3_2021_sales_growth=random.randint(100000, 999999),
            q4_2021_sales_growth=random.randint(100000, 999999),
            total_2021_sales_growth=random.randint(1000000, 9999999),
            q1_2022_rev=random.randint(1000000, 9999999),
            q2_2022_rev=random.randint(1000000, 9999999),
            q3_2022_rev=random.randint(1000000, 9999999),
            q4_2022_rev=random.randint(1000000, 9999999),
            total_2022_rev=random.randint(1000000, 9999999),
            q1_2022_rev_growth=random.randint(100000, 999999),
            q2_2022_rev_growth=random.randint(100000, 999999),
            q3_2022_rev_growth=random.randint(100000, 999999),
            q4_2022_rev_growth=random.randint(100000, 999999),
            total_2022_rev_growth=random.randint(1000000, 9999999),
            q1_2021_rev=random.randint(1000000, 9999999),
            q2_2021_rev=random.randint(1000000, 9999999),
            q3_2021_rev=random.randint(1000000, 9999999),
            q4_2021_rev=random.randint(1000000, 9999999),
            total_2021_rev=random.randint(1000000, 9999999),
            q1_2021_rev_growth=random.randint(100000, 999999),
            q2_2021_rev_growth=random.randint(100000, 999999),
            q3_2021_rev_growth=random.randint(100000, 999999),
            q4_2021_rev_growth=random.randint(100000, 999999),
            total_2021_rev_growth=random.randint(1000000, 9999999),

        )
