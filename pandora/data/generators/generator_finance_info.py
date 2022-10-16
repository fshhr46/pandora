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
    sales_agg: str
    sales_aag: str
    sales_agg_2022: str
    sales_aag_2022: str
    sales_agg_q1: str
    sales_aag_q1: str
    agg_sales: str
    aag_sales: str
    agg_sales_2022: str
    aag_sales_2022: str
    agg_sales_q1: str
    aag_sales_q1: str
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
    total_q1_rev: str
    total_q2_rev: str
    total_q3_rev: str
    total_q4_rev: str
    growth_q1_rev: str
    growth_q2_rev: str
    growth_q3_rev: str
    growth_q4_rev: str
    total_rev_q1: str
    total_rev_q2: str
    total_rev_q3: str
    total_rev_q4: str
    growth_rev_q1: str
    growth_rev_q2: str
    growth_rev_q3: str
    growth_rev_q4: str


class FinanceReportGenerator(DataGeneratorBase):
    def _generate(self):
        return FinanceReport(
            sales_agg=random.randint(10000000, 99999999),
            sales_aag=random.randint(10000000, 99999999),
            sales_agg_2022=random.randint(10000000, 99999999),
            sales_aag_2022=random.randint(10000000, 99999999),
            sales_agg_q1=random.randint(10000000, 99999999),
            sales_aag_q1=random.randint(10000000, 99999999),
            agg_sales=random.randint(10000000, 99999999),
            aag_sales=random.randint(10000000, 99999999),
            agg_sales_2022=random.randint(10000000, 99999999),
            aag_sales_2022=random.randint(10000000, 99999999),
            agg_sales_q1=random.randint(10000000, 99999999),
            aag_sales_q1=random.randint(10000000, 99999999),
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
            total_q1_rev=random.randint(100000, 999999),
            total_q2_rev=random.randint(100000, 999999),
            total_q3_rev=random.randint(100000, 999999),
            total_q4_rev=random.randint(100000, 999999),
            growth_q1_rev=random.randint(100000, 999999),
            growth_q2_rev=random.randint(100000, 999999),
            growth_q3_rev=random.randint(100000, 999999),
            growth_q4_rev=random.randint(100000, 999999),
            total_rev_q1=random.randint(100000, 999999),
            total_rev_q2=random.randint(100000, 999999),
            total_rev_q3=random.randint(100000, 999999),
            total_rev_q4=random.randint(100000, 999999),
            growth_rev_q1=random.randint(100000, 999999),
            growth_rev_q2=random.randint(100000, 999999),
            growth_rev_q3=random.randint(100000, 999999),
            growth_rev_q4=random.randint(100000, 999999),
        )
