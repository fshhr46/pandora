
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
    "gender_digit": ["性别"],

    # =========== 身份信息
    # 邮箱
    "email_add": ["邮箱地址"],
    # 电话号码
    "phone_number": ["电话号码"],
    # 身份证
    "id": ["身份证号码"],
    "id_head_6": ["身份证号码"],
    "cert_type": ["证件类型"],

    # =========== 业务信息
    "acc_open_date": ["开户日期"],
    "is_active_acc": ["活跃账户"],
    "service_type": ["业务类型"],

    # =========== 营收信息
    "q1_2022_rev": ["季度营收"],
    "total_2022_rev": ["年度营收"],
    "q1_2022_rev_growth": ["季度营收增长"],
    "total_2022_rev_growth": ["年度营收增长"],

    # =========== 销量信息
    "q1_2022_sales": ["季度销量"],
    "total_2022_sales": ["年度销量"],
    "q1_2022_sales_growth": ["季度销量增长"],
    "total_2022_sales_growth": ["年度销量增长"],

    # =========== 联系信息
    # 居住地址
    "home_add": ["家庭地址"],
    "jiating_dizhi": ["家庭地址"],
    "home_postal_code": ["家庭地址邮编"],

    # 工作地址
    "work_add": ["工作地址"],
    "gongzuo_dizhi": ["工作地址"],
    "work_postal_code": ["工作地址邮编"],
}


CLASSIFICATION_COLUMN_2_LABEL_ID_TEST = {

    # =========== 个人信息
    # 名字
    "name_cn": ["人名"],
    "name_cn_masked": ["人名"],
    "name_us": ["人名"],
    # 年龄
    "age": ["年龄"],
    # 生日
    "birthday": ["出生日期"],
    # 性别
    "gender": ["性别"],
    "gender_digit": ["性别"],

    # =========== 身份信息
    # 邮箱
    "email_add": ["邮箱地址"],
    "email_add_masked": ["邮箱地址"],
    "data": ["邮箱地址"],

    # 电话号码
    "phone_number": ["电话号码"],
    "phone_number_masked": ["电话号码"],

    # 身份证
    "id": ["身份证号码"],
    "id_masked": ["身份证号码"],
    "id_head_6": ["身份证号码"],
    "cert_type": ["证件类型"],

    # =========== 业务信息
    "acc_open_date": ["开户日期"],
    "is_active_acc": ["活跃账户"],
    "service_type": ["业务类型"],

    # =========== 营收信息
    # 2022
    "q1_2022_rev": ["季度营收"],
    "q2_2022_rev": ["季度营收"],
    "total_2022_rev": ["年度营收"],
    "q1_2022_rev_growth": ["季度营收增长"],
    "q2_2022_rev_growth": ["季度营收增长"],
    "total_2022_rev_growth": ["年度营收增长"],
    # 2021
    "rev_q1_2021": ["季度营收"],
    "rev_q2_2021": ["季度营收"],
    "rev_total_2021": ["年度营收"],
    "rev_growth_q1_2021": ["季度营收增长"],
    "rev_growth_q2_2021": ["季度营收增长"],
    "rev_growth_total_2021": ["年度营收增长"],

    # =========== 销量信息
    # 2022
    "q1_2022_sales": ["季度销量"],
    "q2_2022_sales": ["季度销量"],
    "total_2022_sales": ["年度销量"],
    "q1_2022_sales_growth": ["季度销量增长"],
    "q2_2022_sales_growth": ["季度销量增长"],
    "total_2022_sales_growth": ["年度销量增长"],

    # 2021
    "2021_q1_sales": ["季度销量"],
    "2021_q2_sales": ["季度销量"],
    "2021_total_sales": ["年度销量"],
    "2021_q1_sales_growth": ["季度销量增长"],
    "2021_q2_sales_growth": ["季度销量增长"],
    "2021_total_sales_growth": ["年度销量增长"],

    # =========== 联系信息
    # 居住地址
    "home_add": ["家庭地址"],
    "jiating_dizhi": ["家庭地址"],
    "home_postal_code": ["家庭地址邮编"],

    # 工作地址
    "work_add": ["工作地址"],
    "gongzuo_dizhi": ["工作地址"],
    "work_postal_code": ["工作地址邮编"],
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
    "data": "数据",
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
    "rev_q1_2021": "2021年1季度营收",
    "rev_q2_2021": "2021年2季度营收",
    "total_2022_rev": "2022年全年营收",
    "rev_total_2021": "2021年全年营收",
    "q1_2022_rev_growth": "2022年1季度营收增长",
    "q2_2022_rev_growth": "2022年2季度营收增长",
    "rev_growth_q1_2021": "2021年1季度营收增长",
    "rev_growth_q2_2021": "2021年2季度营收增长",
    "total_2022_rev_growth": "2022年全年营收增长",
    "rev_growth_total_2021": "2021年全年营收增长",

    # =========== 销量信息
    "q1_2022_sales": "2022年1季度销售额",
    "q2_2022_sales": "2022年2季度销售额",
    "2021_q1_sales": "2021年1季度销售额",
    "2021_q2_sales": "2021年2季度销售额",
    "total_2022_sales": "2022年全年销售额",
    "2021_total_sales": "2021年全年销售额",
    "q1_2022_sales_growth": "2022年1季度销售额增长",
    "q2_2022_sales_growth": "2022年2季度销售额增长",
    "2021_q1_sales_growth": "2021年1季度销售额增长",
    "2021_q2_sales_growth": "2021年2季度销售额增长",
    "total_2022_sales_growth": "2022年全年销售额增长",
    "2021_total_sales_growth": "2021年全年销售额增长",

    # =========== 联系信息
    # 居住地址
    "home_add": "居住地址",
    "jiating_dizhi": "居住地址",
    "home_postal_code": "居住地邮编",

    # 工作地址
    "work_add": "工作地址",
    "gongzuo_dizhi": "工作地址",
    "work_postal_code": "工作地邮编",

    # 邮寄地址
    "mail_add": "邮寄地址",
}
