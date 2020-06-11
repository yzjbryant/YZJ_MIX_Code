class Person:
    def cry(self):
        print("I can cry")
    def speak(self):
        print("I can speak %s"%(self.word))

tom=Person()
tom.cry()
tom.word="hahah"
tom.speak()

class Person1:
    def __init__(self):
        self.country="china"
        self.sex="male"
    def speak(self):
        print("I am from %s"%self.country)

jack=Person1()
jack.speak()
print(jack.country,jack.sex)


class Person2:
    def __init__(self,name,age):
        self.age=age
        self.name=name
        self.country="china"
    def speak(self):
        print("name=%s,age=%d"%(self.name,self.age))

p1=Person2("jack",19)
p2=Person2("Tom",22)
p3=p2

p1.speak()
p2.speak()
print(p3)
print(p2)