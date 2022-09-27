
from pandora.data.generators.generator_phone import PhoneNumberGenerator
from pandora.data.generators.generator_name import NameGenerator
from pandora.data.generators.generator_person import PersonalInfoGenerator
from pandora.data.generators.generator_ip import IPGenerator
from pandora.data.generators.generator_edu_info import EduInfoGenerator
from pandora.data.generators.generator_email import EmailAddressGenerator
from pandora.data.generators.generator_finance_info import FinanceInfoGenerator
from pandora.data.generators.generator_address import AddressInfoGenerator


DATA_GENERATORS = [
    PersonalInfoGenerator,
    PhoneNumberGenerator,
    NameGenerator,
    EmailAddressGenerator,
    EduInfoGenerator,
    IPGenerator,
    FinanceInfoGenerator,
    AddressInfoGenerator,
]


CLASSIFICATION_LABELS = [
    # 基于内容
    "人名",
    "电话号码",
    "邮箱地址",
    # 结合内容 + 字段名
    "工作地址",
    "家庭地址",
]

CLASSIFICATION_COLUMN_2_LABEL_ID_TRAIN = {
    "name_cn": ["人名"],
    "email_add": ["邮箱地址"],
    "phone_number": ["电话号码"],
    "home_add": ["家庭地址"],
    "jiating_dizhi": ["家庭地址"],
    "work_add": ["工作地址"],
    "gongzuo_dizhi": ["工作地址"],
}


CLASSIFICATION_COLUMN_2_LABEL_ID_TEST = {
    "name_cn": ["人名"],
    "name_cn_masked": ["人名"],
    # "name_us": ["人名"],
    "email_add": ["邮箱地址"],
    "email_add_masked": ["邮箱地址"],
    "random_col_name": ["邮箱地址"],
    "phone_number": ["电话号码"],
    # "phone_number_formatted": ["电话号码"],
    "phone_number_masked": ["电话号码"],
    "home_add": ["家庭地址"],
    "jiating_dizhi": ["家庭地址"],
    "work_add": ["工作地址"],
    "gongzuo_dizhi": ["工作地址"],
}
