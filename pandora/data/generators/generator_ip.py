import random
from dataclasses import dataclass

from pandora.data.generators.generator_base import DataGeneratorBase
from pandora.data.resource.names import NAMES_CN, NAMES_US


@dataclass
class IPAddresses(object):
    ipv4: str
    ipv6: str
    mac: str


class IPGenerator(DataGeneratorBase):
    def randomMAC(self):
        mac = [0x52, 0x54, 0x00,
               random.randint(0x00, 0x7f),
               random.randint(0x00, 0xff),
               random.randint(0x00, 0xff)]
        return ':'.join(map(lambda x: "%02x" % x, mac))

    def _generate(self):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        m = random.randint(0, 255)
        n = random.randint(0, 255)
        x = random.randint(0, 255)
        y = random.randint(0, 255)
        return IPAddresses(
            f"{m}.{n}.{x}.{y}",
            f"{a}.{b}.{m}.{n}.{x}.{y}",
            self.randomMAC()
        )
