
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
    "个人信息",  # 包括 人名，年龄, 生日，性别(0和1)
    "身份信息",  # 邮箱，电话号码，身份证，身份证前六位(id_head_6)，cert_type(证件类型，0-4)

    # 基于字段名
    "业务信息",  # 账户创建日期，活跃账户(0和1)，套餐类型(service_type, 0-4)
    "营收信息",  # 组合词能力：q1, sales, 2022, rev
    "销量信息",  # 与营收信息混淆.

    # 需基于两者结合 - address，只用元数据的话会与邮箱混淆。
    "联系信息",  # home_add, work_add, home_postal_code, work_postal_code(六位数)
]

CLASSIFICATION_COLUMN_2_LABEL_ID_TRAIN = {
    # =========== 个人信息
    # 名字
    "name_cn": ["个人信息"],
    # 年龄
    "age": ["个人信息"],
    # 生日
    "birthday": ["个人信息"],
    # 性别
    "gender": ["个人信息"],
    "gender_digit": ["个人信息"],

    # =========== 身份信息
    # 邮箱
    "email_add": ["身份信息"],
    # 电话号码
    "phone_number": ["身份信息"],
    # 身份证
    "id": ["身份信息"],
    "id_head_6": ["身份信息"],
    "cert_type": ["身份信息"],

    # =========== 业务信息
    "acc_open_date": ["业务信息"],
    "is_active_acc": ["业务信息"],
    "service_type": ["业务信息"],

    # =========== 营收信息
    "q1_2022_rev": ["营收信息"],
    "q1_2022_rev_growth": ["营收信息"],
    "total_2022_rev_growth": ["营收信息"],

    # =========== 销量信息
    "q1_2022_sales": ["销量信息"],
    "q1_2022_sales_growth": ["销量信息"],
    "total_2022_sales_growth": ["销量信息"],

    # =========== 联系信息
    # 居住地址
    "home_add": ["联系信息"],
    "jiating_dizhi": ["联系信息"],

    # 工作地址
    "gongzuo_dizhi": ["联系信息"],
    "work_postal_code": ["联系信息"],
}


CLASSIFICATION_COLUMN_2_LABEL_ID_TEST = {

    # =========== 个人信息
    # 名字
    "name_cn": ["个人信息"],
    "name_cn_masked": ["个人信息"],
    "name_us": ["个人信息"],
    # 年龄
    "age": ["个人信息"],
    # 生日
    "birthday": ["个人信息"],
    # 性别
    "gender": ["个人信息"],
    "gender_digit": ["个人信息"],

    # =========== 身份信息
    # 邮箱
    "email_add": ["身份信息"],
    "email_add_masked": ["身份信息"],
    "col_1": ["身份信息"],
    # 电话号码
    "phone_number": ["身份信息"],
    "phone_number_masked": ["身份信息"],
    # 身份证
    "id": ["身份信息"],
    "id_masked": ["身份信息"],
    "id_head_6": ["身份信息"],
    "cert_type": ["身份信息"],

    # =========== 业务信息
    "acc_open_date": ["业务信息"],
    "is_active_acc": ["业务信息"],
    "service_type": ["业务信息"],

    # =========== 营收信息
    "q1_2022_rev": ["营收信息"],
    "q2_2022_rev": ["营收信息"],
    "q3_2022_rev": ["营收信息"],
    "q4_2022_rev": ["营收信息"],
    "total_2022_rev": ["营收信息"],
    "q1_2022_rev_growth": ["营收信息"],
    "q2_2022_rev_growth": ["营收信息"],
    "q3_2022_rev_growth": ["营收信息"],
    "q4_2022_rev_growth": ["营收信息"],
    "total_2022_rev_growth": ["营收信息"],
    "q1_2021_rev": ["营收信息"],
    "q2_2021_rev": ["营收信息"],
    "q3_2021_rev": ["营收信息"],
    "q4_2021_rev": ["营收信息"],
    "total_2021_rev": ["营收信息"],
    "q1_2021_rev_growth": ["营收信息"],
    "q2_2021_rev_growth": ["营收信息"],
    "q3_2021_rev_growth": ["营收信息"],
    "q4_2021_rev_growth": ["营收信息"],
    "total_2021_rev_growth": ["营收信息"],

    # =========== 销量信息
    "q1_2022_sales": ["销量信息"],
    "q2_2022_sales": ["销量信息"],
    "q3_2022_sales": ["销量信息"],
    "q4_2022_sales": ["销量信息"],
    "total_2022_sales": ["销量信息"],
    "q1_2022_sales_growth": ["销量信息"],
    "q2_2022_sales_growth": ["销量信息"],
    "q3_2022_sales_growth": ["销量信息"],
    "q4_2022_sales_growth": ["销量信息"],
    "total_2022_sales_growth": ["销量信息"],
    "q1_2021_sales": ["销量信息"],
    "q2_2021_sales": ["销量信息"],
    "q3_2021_sales": ["销量信息"],
    "q4_2021_sales": ["销量信息"],
    "total_2021_sales": ["销量信息"],
    "q1_2021_sales_growth": ["销量信息"],
    "q2_2021_sales_growth": ["销量信息"],
    "q3_2021_sales_growth": ["销量信息"],
    "q4_2021_sales_growth": ["销量信息"],
    "total_2021_sales_growth": ["销量信息"],

    # =========== 联系信息
    # 居住地址
    "home_add": ["联系信息"],
    "jiating_dizhi": ["联系信息"],
    "home_postal_code": ["联系信息"],

    # 工作地址
    "work_add": ["联系信息"],
    "gongzuo_dizhi": ["联系信息"],
    "work_postal_code": ["联系信息"],
}

CLASSIFICATION_COLUMN_2_COMMENT = {

    # =========== 个人信息
    # 名字
    "name_cn": "中文姓名",
    "name_cn_masked": "脱敏中文名",
    "name_us": "英文姓名",
    # 年龄
    "age": "年龄",
    # 生日
    "birthday": "出生日期",
    # 性别
    "gender": "性别",
    "gender_digit": "性别",

    # =========== 身份信息
    # 邮箱
    "email_add": "邮箱地址",
    "email_add_masked": "脱敏邮箱地址",
    "col_1": "字段名1",
    # 电话号码
    "phone_number": "电话号码",
    "phone_number_masked": "脱敏电话号码",
    # 身份证
    "id": "证件号码",
    "id_masked": "脱敏证件号码",
    "id_head_6": "证件号码前六位",
    "cert_type": "证件类型",

    # =========== 业务信息
    "acc_open_date": "开户日期",
    "is_active_acc": "活跃账户",
    "service_type": "套餐类型",

    # =========== 营收信息
    "q1_2022_rev": "2022年1季度营收",
    "q2_2022_rev": "2022年2季度营收",
    "q3_2022_rev": "2022年3季度营收",
    "q4_2022_rev": "2022年4季度营收",
    "total_2022_rev": "2022年全年营收",
    "q1_2022_rev_growth": "2022年1季度营收增长",
    "q2_2022_rev_growth": "2022年2季度营收增长",
    "q3_2022_rev_growth": "2022年3季度营收增长",
    "q4_2022_rev_growth": "2022年4季度营收增长",
    "total_2022_rev_growth": "2022年全年营收增长",
    "q1_2021_rev": "2021年1季度营收",
    "q2_2021_rev": "2021年2季度营收",
    "q3_2021_rev": "2021年3季度营收",
    "q4_2021_rev": "2021年4季度营收",
    "total_2021_rev": "2021年全年营收",
    "q1_2021_rev_growth": "2021年1季度营收增长",
    "q2_2021_rev_growth": "2021年2季度营收增长",
    "q3_2021_rev_growth": "2021年3季度营收增长",
    "q4_2021_rev_growth": "2021年4季度营收增长",
    "total_2021_rev_growth": "2021年全年营收增长",

    # =========== 销量信息
    "q1_2022_sales": "2022年1季度销售额",
    "q2_2022_sales": "2022年2季度销售额",
    "q3_2022_sales": "2022年3季度销售额",
    "q4_2022_sales": "2022年4季度销售额",
    "total_2022_sales": "2022年全年销售额",
    "q1_2022_sales_growth": "2022年1季度销售额增长",
    "q2_2022_sales_growth": "2022年2季度销售额增长",
    "q3_2022_sales_growth": "2022年3季度销售额增长",
    "q4_2022_sales_growth": "2022年4季度销售额增长",
    "total_2022_sales_growth": "2022年全年销售额增长",
    "q1_2021_sales": "2021年1季度销售额",
    "q2_2021_sales": "2021年2季度销售额",
    "q3_2021_sales": "2021年3季度销售额",
    "q4_2021_sales": "2021年4季度销售额",
    "total_2021_sales": "2021年全年销售额",
    "q1_2021_sales_growth": "2021年1季度销售额增长",
    "q2_2021_sales_growth": "2021年2季度销售额增长",
    "q3_2021_sales_growth": "2021年3季度销售额增长",
    "q4_2021_sales_growth": "2021年4季度销售额增长",
    "total_2021_sales_growth": "2021年全年销售额增长",

    # =========== 联系信息
    # 居住地址
    "home_add": "居住地址",
    "jiating_dizhi": "居住地址",
    "home_postal_code": "居住地邮编",

    # 工作地址
    "work_add": "工作地址",
    "gongzuo_dizhi": "工作地址",
    "work_postal_code": "工作地邮编",
}
