# my_classes.py

# from pydantic import UUID4, BaseModel, Field
# import uuid
from ucsbdistrict.basemodel import BaseEquipment


class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"

class AnotherClass:
    def __init__(self, value):
        self.value = value

    def double(self):
        return self.value * 2
    

class MyClassWithValidation(BaseEquipment):
    name: str
