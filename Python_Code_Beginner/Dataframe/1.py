# def sqrt(x):
#     y=1.0
#     while abs(y*y-x)>1e-6:
#         y=(y+x/y)/2
#     return y
#
# def coloring(G):
#     color=0
#     groups=set()
#     verts=vertices(G)
#     while verts:
#         new_group=set()
#         for v in list(verts)
#             if not_adjacent_with_set(v,newgroup,G):
#                 new_group.add(v)
#                 verts.remove(v)
#         groups.add({color,new_group})
#         color +=1
#     return group
#
# def fib(n):
#     if n<2:
#         return 1
#     else:
#         return fib(n-1) +fib(n-2)
#
# def fib(n):
#     f1=f2=1
#     for k in range(1,n):
#         f1,f2=f2,f2+f1
#     return f2
#
# for i in range(n):
#     for j in range(n):
#         x=0.0
#         for k in range(n):
#             x=x+m1[i][k] * m2[k][j]
#         m[i][j]=x
#
# #递归算法得算法模式：
# def recur(n):
#     if n==0:
#         return g(...)
#     somework
#     for i in range(a):
#         x = recur(n/b)
#         somework
#     somework
#
#
# data=[]
# while还有数据:
#     x=下一数据
#     data.insert(0,x) #把新数据加在表的最前面
#
# data=[]
# while还有数据:
#     x=下一数据
#     data.insert(len(data),x)#新数据加在最后or:data.append(x)
#
# def test1(n):
#     lst=[]
#     for i in range(n*10000):
#         lst.append(i)
#         lst=lst+[i]
#     return lst
#
# def test2(n):
#     return [i for i in range(n*10000)]
#
# def test3(n):
#     return list(range(n*10000))

# class Rational0:
#     def __init__(self,num,den):
#         self.num=num
#         self.den=den
#
#     def plus(self,another):
#         den=self.den*another.den
#         num=(self.num*another.den
#              +self.den*another.num)
#         return Rational0(num,den)
#
#     def print(self):
#         print(str(self.num)+"/"+str(self.den))
#
# r1=Rational0(3,5)
# r2=r1.plus(Rational0(7,15))
# r2.print()
#
# class Rational:
#     @staticmethod
#     def _gcd(m,n):
#         if n==0:
#             m,n=n,m
#         while m !=0:
#             m,n=n%m,m
#         return n
#
#     def __init__(self,num,den=1):
#         if not isinstance(num,int) or not isinstance(den,int):
#             raise TypeError
#         if den==0:
#             raise ZeroDivisionError
#         sign=1
#         if num<0:
#             num,sign=-num,-sign
#         if den<0:
#             den,sign=-den,-sign
#         g=Rational._gcd(num,den)
#         self._num=sign*(num//g)
#         self._den=den//g
#
#
#
#
#
#
# def __add__(self,another):
#     den=self._den*another.den()
#     num=(self._num*another.den()+self._den*another.num())
#     return Rational(num,den)
#
# def __mul__(self,another):
#     return Rational(self._num*another.num(),
#                     self._den*another.den())
#
# def __floordiv__(self,another):
#     if another.num()==0:
#         raise ZeroDivisionError
#     return Rational(self._num*another.den(),
#                     self._den*another.num())
#
# def __eq__(self,another):
#     return self._num*another.den() == self._den*another.num()
#
# def __lt__(self,another):
#     return self._num*another.den() < self._den*another.num()
#
# #有理数类的字符串转换方法
# def __str__(self):
#     return str(self._num) + "/" +str(self._den)
# def print(self):
#     print(self._num,"/",self._den)
#
#
# five=Rational(5)
# x=Rational(3,5)
# x.print()
# print("Two thirds are",Rational(2,3))
#
# y =five+x*Rational(5,17)
# if y<Rational(123,11):...
#
# t=type(five)
# if isinstance(five,Rational):...
#
# #类定义的基本语法：
# class <类名>:
#     <语句组>
# x=className()
# twothirds=Rational(3,5)

class Countable:
    counter=0
    def __init__(self):
        Countable.counter +=1
    @classmethod
    def get_count(cls):
        return Countable.counter
x=Countable()
y=Countable()
z=Countable()
print(Countable.get_count())

class C:
    a=0
    b=a+1
X=C.b

class <类名>(BaseClass,...):
    <语句组>

s=MyStr(1234)
issubclass(MyStr,str)
isinstance(a,MyStr)
isinstance(s,str)

class DerivedClass(BaseClass):
    def __init__(self,...):
        BaseClass.__init__(self,...)
        ...... #初始化函数的其他操作
    ......#派生类的其他语句（和函数定义）

class B:
    def f(self):
        self.g()
    def g(self):
        print('B.g called.')

class C(B):
    def g(self):
        print('C.g called.')

super().m(...)

class C1:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def m1(self):
        print(self.x,self.y)
    ......
class C2(C1):
    def m1(self):
        super().m1()
        print("Some special service")
    ......

class RationalError(ValueError):
    pass










