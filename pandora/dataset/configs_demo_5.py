
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
    "人名",
    "年龄",
    "出生日期",
    "性别",

    # 邮箱，电话号码，身份证，身份证前六位(id_head_6)，cert_type(证件类型，0-4)
    "身份证号码",
    "邮箱地址",
    "电话号码",
    "证件类型",

    # 基于字段名
    # 账户创建日期，活跃账户(0和1)业务类型(service_type, 0-4)
    "开户日期",
    "活跃账户",
    "业务类型",

    # 组合词能力：q1, sales, 2022, rev
    # 与营收信息混淆.
    "季度营收",
    "季度营收增长",
    "年度营收",
    "年度营收增长",
    "季度销量",
    "季度销量增长",
    "年度销量",
    "年度销量增长",

    # 需基于两者结合 - address，只用元数据的话会与邮箱混淆。
    # home_add, work_add
    "家庭地址",
    "家庭地址邮编",
    "工作地址",
    "工作地址邮编"
]

CLASSIFICATION_COLUMN_2_LABEL_ID_TRAIN = {
    # =========== 个人信息
    # 名字
    "name_cn": ["人名"],
    # 生日
    "birthday": ["出生日期"],
    # 性别
    "gender": ["性别"],

    # =========== 身份信息
    # 邮箱
    "email_add": ["邮箱地址"],
    # 电话号码
    "phone_number": ["电话号码"],

    # 身份证
    "id": ["身份证号码"],

    # =========== 联系信息
    # 居住地址
    "home_add": ["家庭地址"],
    "home_postal_code": ["家庭地址邮编"],
}


CLASSIFICATION_COLUMN_2_LABEL_ID_TEST = {

    # =========== 个人信息
    # 名字
    "name_cn": ["人名"],
    "name_cn_masked": ["人名"],
    "name_us": ["人名"],
    # 生日
    "birthday": ["出生日期"],
    # 性别
    "gender": ["性别"],

    # =========== 身份信息
    # 邮箱
    "email_add": ["邮箱地址"],
    "email_add_masked": ["邮箱地址"],

    # 电话号码
    "phone_number": ["电话号码"],
    "phone_number_masked": ["电话号码"],

    # 身份证
    "id": ["身份证号码"],
    "id_masked": ["身份证号码"],
    # =========== 联系信息
    # 居住地址
    "home_add": ["家庭地址"],
    "home_postal_code": ["家庭地址邮编"],
}

CLASSIFICATION_COLUMN_2_COMMENT = {

    # =========== 个人信息
    # 名字
    "name_cn": "中文姓名",
    "name_cn_masked": "脱敏中文名",
    "name_us": "英文姓名",
    # 生日
    "birthday": "出生日期",
    # 性别
    "gender": "性别",

    # =========== 身份信息
    # 邮箱
    "email_add": "邮箱地址",
    "email_add_masked": "脱敏邮箱地址",
    # 电话号码
    "phone_number": "电话号码",
    "phone_number_masked": "脱敏电话号码",
    # 身份证
    "id": "证件号码",
    "id_masked": "脱敏证件号码",

    # =========== 联系信息
    # 居住地址
    "home_add": "居住地址",
    "home_postal_code": "居住地邮编",
}


CLASSIFICATION_CLASS_PATHS = [

    # 基于内容
    "/客户/个人/个人自然信息/个人基本概况信息",  # 包括 人名，年龄, 生日，性别(0和1)

    # 邮箱，电话号码，身份证，身份证前六位(id_head_6)，cert_type(证件类型，0-4)
    "/客户/个人/个人身份鉴别信息/传统鉴别信息",

    # 基于字段名
    "/业务/账户信息///基本信息",  # 账户创建日期，活跃账户(0和1)，套餐类型(service_type, 0-4)
    "/经营管理/营销服务/营销信息/营销管理信息/营收信息",  # 组合词能力：q1, sales, 2022, rev
    "/经营管理/营销服务/营销信息/营销管理信息/销量信息",  # 与营收信息混淆.

    # 需基于两者结合 - address，只用元数据的话会与邮箱混淆。
    # home_add, work_add, home_postal_code, work_postal_code(六位数)
    "/客户/个人/个人自然信息/个人地理位置信息",
]

CLASSIFICATION_CLASS_PATH_2_RATING = {
    "/客户/个人/个人自然信息/个人基本概况信息": "2级数据",
    "/客户/个人/个人身份鉴别信息/传统鉴别信息": "3级数据",
    "/业务/账户信息///基本信息": "2级数据",
    "/经营管理/营销服务/营销信息/营销管理信息/营收信息": "1级数据",
    "/经营管理/营销服务/营销信息/营销管理信息/销量信息": "1级数据",
    "/客户/个人/个人自然信息/个人地理位置信息": "2级数据",
}


CLASSIFICATION_RATING_COLUMN_2_CLASSPATH__TRAIN = {
    # =========== 个人信息
    # 名字
    "name_cn": "/客户/个人/个人自然信息/个人基本概况信息",
    # 年龄
    "age": "/客户/个人/个人自然信息/个人基本概况信息",
    # 生日
    "birthday": "/客户/个人/个人自然信息/个人基本概况信息",
    # 性别
    "gender": "/客户/个人/个人自然信息/个人基本概况信息",
    "gender_digit": "/客户/个人/个人自然信息/个人基本概况信息",

    # =========== 身份信息
    # 邮箱
    "email_add": "/客户/个人/个人身份鉴别信息/传统鉴别信息",
    # 电话号码
    "phone_number": "/客户/个人/个人身份鉴别信息/传统鉴别信息",
    # 身份证
    "id": "/客户/个人/个人身份鉴别信息/传统鉴别信息",
    "id_head_6": "/客户/个人/个人身份鉴别信息/传统鉴别信息",
    "cert_type": "/客户/个人/个人身份鉴别信息/传统鉴别信息",

    # =========== 业务信息
    "acc_open_date": "/业务/账户信息///基本信息",
    "is_active_acc": "/业务/账户信息///基本信息",
    "service_type": "/业务/账户信息///基本信息",

    # =========== 营收信息
    "q1_2022_rev": "/经营管理/营销服务/营销信息/营销管理信息/营收信息",
    "total_2022_rev": "/经营管理/营销服务/营销信息/营销管理信息/营收信息",
    "q1_2022_rev_growth": "/经营管理/营销服务/营销信息/营销管理信息/营收信息",
    "total_2022_rev_growth": "/经营管理/营销服务/营销信息/营销管理信息/营收信息",

    # =========== 销量信息
    "q1_2022_sales": "/经营管理/营销服务/营销信息/营销管理信息/销量信息",
    "total_2022_sales": "/经营管理/营销服务/营销信息/营销管理信息/销量信息",
    "q1_2022_sales_growth": "/经营管理/营销服务/营销信息/营销管理信息/销量信息",
    "total_2022_sales_growth": "/经营管理/营销服务/营销信息/营销管理信息/销量信息",

    # =========== 联系信息
    # 居住地址
    "home_add": "/客户/个人/个人自然信息/个人地理位置信息",
    "jiating_dizhi": "/客户/个人/个人自然信息/个人地理位置信息",
    "home_postal_code": "/客户/个人/个人自然信息/个人地理位置信息",

    # 工作地址
    "work_add": "/客户/个人/个人自然信息/个人地理位置信息",
    "gongzuo_dizhi": "/客户/个人/个人自然信息/个人地理位置信息",
    "work_postal_code": "/客户/个人/个人自然信息/个人地理位置信息",
}
