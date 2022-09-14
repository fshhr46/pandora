
from faker import Faker


# https://ost.51cto.com/posts/379

def get_faker(locales=['zh_CN']):
    return Faker(locale=locales)
