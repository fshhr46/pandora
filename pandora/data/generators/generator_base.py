import pandora.data.faker_utils as faker_utils


class DataGeneratorBase(object):

    def __init__(self, masking_func=None, locales=['zh_CN']) -> None:
        self.masking_func = masking_func
        self.faker = faker_utils.get_faker(locales)

    def generate(self):
        data = self._generate()
        if self.masking_func:
            data = self.masking_func(data)
        return data

    def _generate(self):
        raise NotImplementedError
