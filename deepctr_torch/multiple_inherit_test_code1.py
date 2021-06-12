class Father(object):
    def __init__(self, name):
        self.name = name
        print("Im father")


class Son_1(Father):
    def __init__(self, name, age, gender):
        self.age = age
        super(Son_1, self).__init__(name, gender)
        print(Son_1.__mro__)
        print("Im Son_1")


class Son_2(Father):
    def __init__(self, name, gender):
        self.gender = gender
        super(Son_2, self).__init__(name)
        print("我是Son_2")


class GrandSon(Son_1, Son_2):
    def __init__(self, name, age, gender):
        super(GrandSon, self).__init__(name, age, gender)
        print("我是GrandSon")
grand_son = GrandSon("张三", 19, "男",)
print(GrandSon.__mro__)