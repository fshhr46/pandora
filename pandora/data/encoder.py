import json

from pandora.data.generators.generator_person import Person
from pandora.data.generators.generator_phone import PhoneNumber


class DataJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Person):
            return o.__dict__
        if isinstance(o, PhoneNumber):
            return o.__dict__
        return DataJSONEncoder(self, o)
