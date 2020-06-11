# one=1
# two=2
# some_number=10000
# true_boolean=True
# false_boolean=False
# my_name="Leandro TK"
# book_price=15.80
# if True:
#     print("Hello Python if")
# if 2>1:
#     print("2 is greater than 1")
# if 1>2:
#     print("1 is greater than 2")
# elif 2>1:
#     print("1 is not greater than2")
# else:
#     print("1 is equal to 2")
#
# num=1
# while num<=10:
#     print(num)
#     num+=1
# loop_condition=True
# while loop_condition:
#     print("Loop Condition keeps: %s" %(loop_condition) )
#     loop_condition=False
#
# for i in range(1,11):
#     print(i)
# for i in range(10):
#     print(i)
#
# my_integers=[1,2,3,4,5]
# print(my_integers[0])
# print("-------")
# print(my_integers[3])
#
# relatives_names=[
#     "Toshiaki",
#     "Juliana",
#     "Yuji",
#     "Bruno",
#     "Kaio"
# ]
# print(relatives_names[2])
#
# bookshelf=[]
# bookshelf.append("The Effective Engineer")
# bookshelf.append("The 4 Hour Work Week")
# print(bookshelf)
# print("--------")
# print(bookshelf[0])
#
# dictionary_example={
#     "key1" : "value1",
#     "key2" : "value2",
#     "key3" : "value3"
# }
# print(dictionary_example)
#
# dictionary_tk={
#     "name":"Leandro",
#     "nickname":"YK",
#     "nationality":"Brazilian",
#     "age":24
# }
# print("My name is %s"%(dictionary_tk["name"]))
# print("-----------")
# print("My 2name is %s "%(dictionary_tk["nickname"]))
# print("My age is %i and %s"%(dictionary_tk["age"],dictionary_tk["name"]))
# print("-------------")
# for i,j in dictionary_tk.items():
#     print("My %s is %s"%(i,j))
#
# dictionary_tk['agew']=35
# print(dictionary_tk)
#
# bookshelf=[
#     "The Effective Engineer",
#     "The 4 hour work",
#     "Zeros to ine"
# ]
# for book in bookshelf:
#     print(book)
#
# print("000000000000000")
#
# dictionary={"some_key":"sone_value"}
# for i in dictionary:
#     print("%s --> %s"%(i,dictionary[i]))
# for key,value in dictionary.items():
#     print("%s --> %s"%(key,value))
#
# class Vehicle1():
#     pass
#
# class Vehicle():
#     def __init__(self,number_of_wheels,type_of_tank,seating_capacity,maxium_velocity):
#         self.number_of_wheels=number_of_wheels
#         self.type_of_tank=type_of_tank
#         self.seating_capacity=seating_capacity
#         self.maxium_velocity=maxium_velocity
#
#     def number_of_wheels(self):
#         return self.number_of_wheels
#     def set_number_of_wheels(self,number):
#         self.number_of_wheels= number
#
#     def  seat_capacity(self):
#         return self.seat_capacity
#     def set_seat_capacity(self,capacity):
#         self.seating_capacity=capacity
#
#     def maximum_velocity(self):
#         return self.maxium_velocity
#     def set_maximum_velocity(self,velocity):
#         self.maxium_velocity=velocity
#
#
#     def make_noise(self):
#         print('VRUUUUUUUM')
#
# tesla_model_s=Vehicle(4,'electric',5,250)
# tesla_model_s.make_noise()
# tesla_model_s=Vehicle(4,'electric',5,250)
# print(tesla_model_s.number_of_wheels)
# tesla_model_s.number_of_wheels=2
# print(tesla_model_s.number_of_wheels)
# tesla_model_s.seating_capacity=5
# print(tesla_model_s.seating_capacity)
# tesla_model_s.maxium_velocity=6
# print(tesla_model_s.maxium_velocity)
#
# # class Person:
# #     def __init__(self,first_name):
# #         self.first_name=first_name
# # tk=Person('TK')
# # # print(tk.first_name)
# #
# # # class Person1:
# # #     first_name='YK'
# # tk.first_name='Ka'
# # print(tk.first_name)
#
# class  Person:
#     def __init__(self,first_name,email):
#         self.first_name=first_name
#         self._email=email
#
#     def update_email(self,new_email):
#         self._email=new_email
#
#     def email(self):
#         return self._email
#
#
# tk=Person('TK','tk@mail.com')
# print(tk._email)
# tk._email='New_tk@mail.com'
# print(tk.email())
# tk.update_email('Nrerdfdfd')
# print(tk.email())
#
#
# # class P:
# #     def __init__(self,first_name,age):
# #         self.first_name=first_name
# #         self._age=age
# #     def show_age(self):
# #         return self._age
# # tk=P('YK',25)
# # print(tk.show_age())
#
# class P:
#     def __init__(self,first_name,age):
#         self.first_name=first_name
#         self._age=age
#     def _show_age(self):
#         return self._age
# tk=P('YK',25)
# print(tk._show_age())
# print("______________________-")
#
# class Person:
#     def __init__(self,first_name,age):
#         self.first_name=first_name
#         self._age=age
#     def show_age(self):
#         return self._get_age()
#     def _get_age(self):
#         return self._age
# tk=Person('TK',25)
# print(tk.show_age())
# print("-------------------------")

class Car:
    def __init__(self,number_of_wheels,seating_capacity,maximum_velocity):
        self.number_of_wheels=number_of_wheels
        self.seating_capacity=seating_capacity
        self.maximum_velocity=maximum_velocity
my_car=Car(4,5,250)
print(my_car.number_of_wheels)
print(my_car.seating_capacity)
print(my_car.maximum_velocity)
print("--------------")
class ElectricCar(Car):
    def __init__(self,number_of_wheels,seating_capacity,maximum_velocity):
        Car.__init__(self,number_of_wheels,seating_capacity,maximum_velocity)
my_electric_car=ElectricCar(6,7,2250)
print(my_electric_car.number_of_wheels)
print(my_electric_car.seating_capacity)
print(my_electric_car.maximum_velocity)




