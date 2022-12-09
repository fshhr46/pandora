
from pandora.data.generators.generator_phone import PhoneNumberGenerator
from pandora.data.generators.generator_name import NameGenerator
from pandora.data.generators.generator_person import PersonalInfoGenerator
from pandora.data.generators.generator_ip import IPGenerator
from pandora.data.generators.generator_edu_info import EduInfoGenerator
from pandora.data.generators.generator_email import EmailAddressGenerator
from pandora.data.generators.generator_finance_info import FinanceInfoGenerator, FinanceReportGenerator
from pandora.data.generators.generator_operation_info import OperationInfoGenerator
from pandora.data.generators.generator_address import AddressInfoGenerator


DATA_GENERATORS = [
    PersonalInfoGenerator,
    PhoneNumberGenerator,
    NameGenerator,
    EmailAddressGenerator,
    EduInfoGenerator,
    IPGenerator,
    FinanceInfoGenerator,
    FinanceReportGenerator,
    OperationInfoGenerator,
    AddressInfoGenerator,
]


CLASSIFICATION_LABELS = [
    # 基于内容
    # 包括 人名，年龄, 生日，性别(0和1)
    "姓名",
    "脱敏姓名",
    "邮箱",
    "脱敏邮箱",
    "地址",
]

CLASSIFICATION_COLUMN_2_LABEL_ID_TRAIN = {
    # 名字
    "name_cn": ["姓名"],
    "name_cn_masked": ["脱敏姓名"],

    # 邮箱
    "email_add": ["邮箱"],
    "email_add_masked": ["脱敏邮箱"],

    # 地址
    "home_add": ["地址"],
}


CLASSIFICATION_COLUMN_2_LABEL_ID_TEST = {
    # 名字
    "name_cn": ["姓名"],
    "name_cn_masked": ["脱敏姓名"],

    # 邮箱
    "email_add": ["邮箱"],
    "email_add_masked": ["脱敏邮箱"],

    # 地址
    "home_add": ["地址"],
}

CLASSIFICATION_COLUMN_2_COMMENT = {

    # =========== 个人信息
    # 名字
    "name_cn": "姓名",
    "name_cn_masked": "脱敏姓名",

    "email_add": "邮箱",
    "email_add_masked": "脱敏邮箱",

    "home_add": "地址",
}


CLASSIFICATION_CLASS_PATHS = [

    # 基于内容
    "/客户/个人/个人自然信息/个人基本概况信息",  # 包括 人名，年龄, 生日，性别(0和1)

    # 邮箱，电话号码，身份证，身份证前六位(id_head_6)，cert_type(证件类型，0-4)
    "/客户/个人/个人身份鉴别信息/传统鉴别信息",

    # 需基于两者结合 - address，只用元数据的话会与邮箱混淆。
    # home_add, work_add, home_postal_code, work_postal_code(六位数)
    "/客户/个人/个人自然信息/个人地理位置信息",
]

CLASSIFICATION_CLASS_PATH_2_RATING = {
    "/客户/个人/个人自然信息/个人基本概况信息": "2级数据",
    "/客户/个人/个人身份鉴别信息/传统鉴别信息": "3级数据",
    "/客户/个人/个人自然信息/个人地理位置信息": "1级数据",
}


CLASSIFICATION_RATING_COLUMN_2_CLASSPATH__TRAIN = {
    # =========== 个人信息
    # 名字
    "name_cn": "/客户/个人/个人自然信息/个人基本概况信息",
    "name_cn_masked": "/客户/个人/个人自然信息/个人基本概况信息",
    # =========== 身份信息
    # 邮箱
    "email_add": "/客户/个人/个人身份鉴别信息/传统鉴别信息",
    "email_add_masked": "/客户/个人/个人身份鉴别信息/传统鉴别信息",

    # 居住地址
    "home_add": "/客户/个人/个人自然信息/个人地理位置信息",
}
