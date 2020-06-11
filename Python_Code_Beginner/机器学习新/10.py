# # x=1==1 or 2!=1
# # print(x)
# #
# # if 1!=2:
# #     print("d")
# # else: print("c")
# #
# # dog=5
# # dog -=5
# # print(dog)
# #
# # people=10
# # ca=2
# # sds=1
# # if people<ca:
# #     print("dsds")
# # elif ca<sds:
# #     print("dsdsds")
# #
# # door=input(">")
# # if door=="1":
# #     print("dds")
# # else: print("dsdsds")
# #
# # the_count=[1,2,3,4]
# # for i in the_count:
# #     print(f"dasd{i}")
# #
# # element=[]
# # for i in range(9):
# #     print(f"asas{i}")
# #     element.append(i)
# # for i in element:
# #     print(i)
# #
# # x=[[1,2,3],[4,5,6]]=inpu
# # print(x)
# #
# #
#
# # ten_things="dsds sdsds dsdsds sdsds "
# # print("sddadasdasdasda")
# # stuff=ten_things.split('')
# # more_stuff=["dad","sadasd"]
# # while len(stuff) !=10:
# #     next_one=more_stuff.pop()
# #     stuff.append(next_one)
# #     print(next_one)
# #     print(len(stuff))
#
# # mystuff={'sas':"sdasdsa"}
# # print(mystuff['sas'])
# # def apple():
# #     print("adadas")
# # import mystuff
# # mystuff.apple()
# #
# # class Mystuff(object):
# #     def __init__(self):
# #         self.tangerine="And dasdasda"
# #     def apple(self):
# #         print("dsdsd")
# # thing=Mystuff()
# # thing.apple()
# # print(thing.tangerine)
# #
# # class X(Y):
# #     def __init__(self,J):
# #     def f(self):
# #     foo=X
# #     foo.M(J)
# #     foo.K=Q
# #
#
#
# import random
# from urllib.request import urlopen
# import sys
#
# WORD_URL="http://learncodethehardway.org/word.txt"
# WORDS=[]
#
# PHRASES={
#     "class %%%(%%%):":
#         "Make a class named %% that is-a %%%.",
#     "class %%%(object):\n\tdef __init__(self,***)":
#         "class %%% has-a __init__ that takes self and *** params.",
#     "class %%%(object):\n\tdef ***(self,@@@)":
#         "class %%% has-a function *** that takes self and @@@ params.",
#     "***=%%%()":
#         "Set *** to an instance of class %%%.",
#     "***.***(@@@)":
#         "From *** get the *** function, call it with params self, @@@.",
#     "***.***='***":
#         "From *** get the *** attribute and set it to '***'."
# }
#
# if len(sys.argv) == 2 and sys.argv[1] =="english":
#     PHRASES_FIRST=True
# else:
#     PHRASES_FIRST=False
#
# for word in urlopen(WORD_URL).readlines():
#     WORDS.append(str(word.strip(),encoding="utf-8"))
#
# def convert(snippet,phrase):
#     class_names=[w.capitalize() for w in
#                  random.sample(WORDS,snippet.count("%%%"))]
#     other_names=random.sample(WORDS,snippet.count("***"))
#     results=[]
#     param_names=[]
#
#     for i in range(0,snippet.count("@@@")):
#         param_count=random.randint(1,3)
#         param_count.append(', '.join(
#             random.sample(WORDS,param_count)
#         ))
#
#     for sentence in snippet, phrase:
#         result=sentence[:]
#
#         for word in class_names:
#             result=result.replace("%%%",word,1)
#         for word in other_names:
#             result=result.replace("***",word,1)
#         for word in param_names:
#             result=result.replace("@@@",word,1)
#
#         results.append(result)
#     return results
#
# try:
#     while True:
#         snippets=list(PHRASES.keys())
#         random.shuffle(snippets)
#
#         for snippet in snippets:
#             phrase=PHRASES[snippet]
#             question,answer=convert(snippet,phrase)
#             if PHRASES_FIRST:
#                 question,answer=answer,question
#             print(question)
#
#             input("> ")
#             print(f"ANSWER: {answer}\n\n")
# except EOFError:
#     print("\nBye")
#
# class Animals(object):
#     pass
#
# class Dog(object):
#     def __init__(self,name):
#         self.name=name
# class Cat(object):
#     def __init__(self,name):
#         self.name=name
# class Person(object):
#     def __init__(self,name):
#         self.name=name
#         self.pet=one
#
# class Employee(Person):
#     def __init__(self,name,salary):
#         super(Employee,self).__init__(name)
#         self.salary=salary
#
# class Fish(object):
#     pass
# class Salmon(Fish):
#     pass
# class Halibut(Fish):
#     pass
# rover=Dog("Rover")
# satan=Cat("Satan")
# mary=Person("Mary")
# mary.pet=satan
# frank=Employee("Frank",120000)
# frank.pet=rover
# flipper=Fish()
# crouse=Salmon()
# harry=Halibut()
#
#
# class Bag:

class Bag():
    def __init__(self):
        self.data=[]
    def add(self,x):
        for j in range(10):
            self.data.append(x)
    def addmore(self,x):
        for i in range(2):
            self.add(x)
b=Bag()
print(b.data)
b.add('1')
print(b.data)
b.addmore('2')
print(b.data)
