#用Numpy实现K-nearest neighbours回归或分类
def wineprice(rating,age):
    peak_age=rating-50
    price=rating/2
    if age>peak_age:
        price=price*(5-(age-peak_age))
    else:
        price=price*(5*((age+1)/peak_age))
    if price<0:
        price=0
    return price

import HTTP
