
from pandora.data.generators.generator_phone import PhoneNumberGenerator, PhoneNumber
from pandora.data.generators.generator_name import NameGenerator, Name
from pandora.data.generators.generator_person import PersonalInfoGenerator, Person
from pandora.data.generators.generator_ip import IPGenerator, IPAddresses
from pandora.data.generators.generator_edu_info import EduInfoGenerator, EduInfo
from pandora.data.generators.generator_email import EmailAddressGenerator, EmailAddress
from pandora.data.generators.generator_finance_info import FinanceInfoGenerator, FinanceInfo
from pandora.data.generators.generator_address import AddressInfoGenerator, AddressInfo


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

DATA_CLASSES = [
    PhoneNumber,
    Name,
    Person,
    IPAddresses,
    EduInfo,
    EmailAddress,
    FinanceInfo,
    AddressInfo,
]

CLASSIFICATION_LABELS = [
    "身份信息",
    "个人信息",
    "虚拟身份信息",
    "教育工作信息",
    "金融信息",
    "其他信息",
]

CLASSIFICATION_COLUMN_2_LABEL_ID = {
    "age": ["个人信息"],
    "birth_place_code": ["个人信息"],
    "birth_place_name": ["个人信息"],
    "birthday": ["个人信息"],
    "birthday_chars": ["个人信息"],
    "birthday_mdy": ["个人信息"],
    "birthday_slash": ["个人信息"],
    "card_cvc": ["金融信息"],
    "card_expir_date": ["金融信息"],
    "card_num": ["金融信息"],
    "card_num_formatted": ["金融信息"],
    "card_num_masked": ["金融信息"],
    "check_digit": ["其他信息"],
    "email": ["身份信息"],
    "email_masked": ["身份信息"],
    "gender": ["个人信息"],
    "id": ["身份信息"],
    "id_masked": ["身份信息"],
    "ipv4": ["虚拟身份信息"],
    "ipv6": ["虚拟身份信息"],
    "issuer": ["金融信息"],
    "living_address": ["个人信息"],
    "mac": ["虚拟身份信息"],
    "major": ["教育工作信息"],
    "name_cn": ["身份信息"],
    "name_cn_masked": ["身份信息"],
    "name_us": ["身份信息"],
    "phone_number": ["身份信息"],
    "phone_number_formatted": ["身份信息"],
    "phone_number_masked": ["身份信息"],
    "university": ["教育工作信息"]
}
