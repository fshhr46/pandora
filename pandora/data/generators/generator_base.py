import pandora.data.faker_utils as faker_utils


class DataGeneratorBase(object):

    def __init__(self, masking_func=None, locales=['zh_CN'], is_test_data=False) -> None:
        self.masking_func = masking_func
        self.faker = faker_utils.get_faker(locales)
        self.is_test_data = is_test_data

    def generate(self):
        if self.is_test_data:
            data = self._generate_test()
        else:
            data = self._generate()
        if self.masking_func:
            data = self.masking_func(data)
        return data

    def _generate(self):
        raise NotImplementedError

    def _generate_test(self, ):
        return self._generate()
