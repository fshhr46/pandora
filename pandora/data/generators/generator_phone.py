'''
搜集到以下手机号码, 当然这也不全, 不过也可以分析出一些规律了
中国电信号段: 133, 153,  180, 181, 189, 170, 173,  177, 149
中国联通号段: 130, 131, 132, 155, 156, 185, 186, 145, 175, 176, 185, 171
中国移动号段: 134, 135, 136, 137, 138, 139, 150, 151, 152, 158, 159, 182, 183, 184, 172, 147, 178
# 规律总结
第一位永远是  1
第二位可以是  3, 4, 5, 7, 8
第三位是由第二位决定的, 有以下情况: 
13 + 【0-9】
14 + 【5, 7, 9】
15 + 【0-9】 ! 4
17 + 【0-9】! 4and9
18 + 【0-9】
后八位: 是0-9随机
    
'''
from dataclasses import dataclass
import random

from pandora.data.generators.generator_base import DataGeneratorBase
import pandora.data.mask_utils as mask_utils


@dataclass
class PhoneNumber(object):
    phone_number: str
    phone_number_formatted: str
    phone_number_masked: str


class PhoneNumberGenerator(DataGeneratorBase):

    def _generate(self):
        second = random.choice([3, 4, 5, 7, 8])
        third = {
            3: random.randint(0, 9),
            4: random.choice([5, 7, 9]),
            5: random.choice([i for i in range(10) if i != 4]),
            7: random.choice([i for i in range(9) if i != 4]),
            8: random.randint(0, 9),
        }[second]

        last = "".join(str(random.randint(0, 9)) for i in range(8))
        phone_number = "1{}{}{}".format(second, third, last)
        phone_number_formatted = "1{}{} {}".format(second, third, last)
        phone_number_masked = mask_utils.mask_data(
            phone_number, positions=range(3, 7))
        return PhoneNumber(
            phone_number=phone_number,
            phone_number_formatted=phone_number_formatted,
            phone_number_masked=phone_number_masked
        )

    def _generate_test(self):
        second = random.choice([3, 4, 5, 7, 8])
        third = {
            3: random.randint(0, 9),
            4: random.choice([5, 7, 9]),
            5: random.choice([i for i in range(10) if i != 4]),
            7: random.choice([i for i in range(9) if i != 4]),
            8: random.randint(0, 9),
        }[second]

        mask_char = random.choice(mask_utils.get_mask_chars())

        start = random.randint(0, 4)
        rand_len = random.randint(2, 5)
        random_positions = range(start, start + rand_len)

        last = "".join(str(random.randint(0, 9)) for i in range(8))
        phone_number = "1{}{}{}".format(second, third, last)
        phone_number_formatted = "1{}{} {}".format(second, third, last)

        phone_number_masked = mask_utils.mask_data(
            phone_number, mask_char=mask_char, positions=random_positions)
        return PhoneNumber(
            phone_number=phone_number,
            phone_number_formatted=phone_number_formatted,
            phone_number_masked=phone_number_masked
        )
