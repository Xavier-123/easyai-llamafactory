

class FieldIsEmptyError(Exception):  # 继承自 Exception
    def __init__(self, field):
        self.field = field

    def __str__(self):
        return f"{self.field} is empty."