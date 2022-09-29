
from pandora.data.generators.generator_phone import PhoneNumberGenerator
from pandora.data.generators.generator_name import NameGenerator
from pandora.data.generators.generator_person import PersonalInfoGenerator
from pandora.data.generators.generator_ip import IPGenerator
from pandora.data.generators.generator_edu_info import EduInfoGenerator
from pandora.data.generators.generator_email import EmailAddressGenerator
from pandora.data.generators.generator_finance_info import FinanceInfoGenerator, FinanceReportGenerator
from pandora.data.generators.generator_address import AddressInfoGenerator
from pandora.data.generators.generator_age import AgeGenerator


DATA_GENERATORS = [
    PersonalInfoGenerator,
    PhoneNumberGenerator,
    NameGenerator,
    EmailAddressGenerator,
    EduInfoGenerator,
    IPGenerator,
    FinanceInfoGenerator,
    AddressInfoGenerator,
    AgeGenerator,
    FinanceReportGenerator,
]


CLASSIFICATION_LABELS = [
    "用户身份信息",
    "年龄信息",
    "销量信息",
    "营收信息",
]

CLASSIFICATION_COLUMN_2_LABEL_ID_TRAIN = {
    "age": ["年龄信息"],
    "sales_agg": ["销量信息"],
    "q1_rev_total": ["营收信息"],
    "home_add": ["用户身份信息"],
    "work_phone_number": ["用户身份信息"],
}


CLASSIFICATION_COLUMN_2_LABEL_ID_TEST = {
    "age": ["年龄信息"],
    "age1": ["年龄信息"],
    "age2": ["年龄信息"],
    "age3": ["年龄信息"],
    "sales_agg": ["销量信息"],
    "sales_aag": ["销量信息"],
    "sales_agg": ["销量信息"],
    "sales_aag": ["销量信息"],
    "sales_agg_2022": ["销量信息"],
    "sales_aag_2022": ["销量信息"],
    "sales_agg_q1": ["销量信息"],
    "sales_aag_q1": ["销量信息"],
    "agg_sales": ["销量信息"],
    "aag_sales": ["销量信息"],
    "agg_sales_2022": ["销量信息"],
    "aag_sales_2022": ["销量信息"],
    "agg_sales_q1": ["销量信息"],
    "aag_sales_q1": ["销量信息"],
    "q1_rev_total": ["营收信息"],
    "q2_rev_total": ["营收信息"],
    "q3_rev_total": ["营收信息"],
    "q4_rev_total": ["营收信息"],
    "q1_rev_growth": ["营收信息"],
    "q2_rev_growth": ["营收信息"],
    "q3_rev_growth": ["营收信息"],
    "q4_rev_growth": ["营收信息"],
    "home_add": ["用户身份信息"],
    "work_add": ["用户身份信息"],
    "home_phone_number": ["用户身份信息"],
    "work_phone_number": ["用户身份信息"],
}
