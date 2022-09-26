#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version 2.7.13 or 3.7.2

import random
import re
# 导入某个模块的部分类或方法
from datetime import datetime, timedelta
from dataclasses import dataclass

# 导入常量并重命名
import pandora.data.constant as const
from pandora.data.mask_utils import mask_data
from pandora.data.resource.districts import AREA_INFO
from pandora.data.generators.generator_base import DataGeneratorBase


class PersonalInfoGenerator(DataGeneratorBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.id_pool = set()

    @classmethod
    def generate_id(cls, sex=0):
        """随机生成身份证号，sex = 0表示女性，sex = 1表示男性"""

        # 随机生成一个区域码(6位数)
        id_number = str(random.choice(list(AREA_INFO.keys())))
        # 限定出生日期范围(8位数)
        start, end = datetime.strptime(
            "1960-01-01", "%Y-%m-%d"), datetime.strptime("2000-12-30", "%Y-%m-%d")
        birth_days = datetime.strftime(
            start + timedelta(random.randint(0, (end - start).days + 1)), "%Y%m%d")
        id_number += str(birth_days)
        # 顺序码(2位数)
        id_number += str(random.randint(10, 99))
        # 性别码(1位数)
        id_number += str(random.randrange(sex, 10, step=2))
        # 校验码(1位数)
        return id_number + str(cls.get_check_digit(id_number))

    @classmethod
    def verify_id(cls, id_number):
        """校验身份证是否正确"""
        if re.match(const.ID_NUMBER_18_REGEX, id_number):
            check_digit = cls.get_check_digit(id_number)
            return str(check_digit) == id_number[-1]
        else:
            return bool(re.match(const.ID_NUMBER_15_REGEX, id_number))

    @classmethod
    def get_check_digit(cls, id_number):
        """通过身份证号获取校验码"""
        check_sum = 0
        for i in range(0, 17):
            check_sum += ((1 << (17 - i)) % 11) * int(id_number[i])
        check_digit = (12 - (check_sum % 11)) % 11
        return check_digit if check_digit < 10 else 'X'

    def _generate(self):
        random_geneder = random.randint(0, 1)  # 随机生成男(1)或女(0)

        while True:
            id_number = self.generate_id(random_geneder)  # 随机生成身份证号
            if self.id_pool and id_number in self.id_pool:
                continue
            check_id = self.get_check_digit(id_number)
            self.id_pool.add(id_number)
            break
        assert self.verify_id(id_number)  # 检验身份证是否正确:False
        p = Person(id_number, check_id)
        return p


@dataclass
class Person(object):
    def __init__(self,
                 id: str,
                 check_digit: str) -> None:
        self.id = id
        self.id_masked = mask_data(id, range(len(id) - 6, len(id)))

        self.birth_year = int(self.id[6:10])
        self.birth_month = int(self.id[10:12])
        self.birth_day = int(self.id[12:14])

        # public attributes
        self.check_digit = str(check_digit)  # 校验码:7
        self.birth_place_code = str(int(self.id[0:6]))  # 地址编码:410326
        self.birth_place_name = self.get_area_name()  # 地址:河南省洛阳市汝阳县

        # 生日:1995-7-10
        birthday = datetime.strptime(self.get_birthday(), "%Y-%m-%d")
        self.birthday = birthday.strftime("%Y-%m-%d")
        self.birthday_mdy = birthday.strftime("%m-%d-%Y")
        self.birthday_slash = birthday.strftime("%m/%d/%Y")
        self.birthday_chars = birthday.strftime("%B %d, %Y")

        self.age = str(self.get_age())  # 年龄:23(岁)
        self.gender = self.get_gender()  # 性别:1(男)

    # basics
    id: str
    id_masked: str
    check_digit: str

    # personal
    gender: str
    age: str

    # birthday and variations
    birthday: str
    birthday_mdy: str
    birthday_slash: str
    birthday_chars: str

    # birth_place
    birth_place_code: str
    birth_place_name: str

    def get_area_name(self):
        """根据区域编号取出区域名称"""
        return AREA_INFO[str(self.birth_place_code)]

    def get_birthday(self):
        """通过身份证号获取出生日期"""
        return "{0}-{1}-{2}".format(self.birth_year, self.birth_month, self.birth_day)

    def get_age(self):
        """通过身份证号获取年龄"""
        now = (datetime.now() + timedelta(days=1))
        year, month, day = now.year, now.month, now.day

        if year == self.birth_year:
            return 0
        else:
            if self.birth_month > month or (self.birth_month == month and self.birth_day > day):
                return year - self.birth_year - 1
            else:
                return year - self.birth_year

    def get_gender(self):
        """通过身份证号获取性别， 女生: 0, 男生: 1"""
        return "男" if int(self.id[16:17]) % 2 else "女"
