from flask import Flask:
app = Flask(__name__)
app.run(port=5000,debug=True)

#开启Web服务器引擎
$ python flask1.py
* Running on http://127.0.0.1:5000/
* Restarting with reloader

from flask import Flask

app=Flask(__name__)
@app.route('/')
def home():
    return "It's alive!"

app.run(debug=True)

#开启服务器
$ python flask2.py
* Running on http://127.0.0.1:5000/
* Restarting with reloader

#进入主页
$ curl http://localhost:5000/
It's alive!



