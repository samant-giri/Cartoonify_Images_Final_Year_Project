from flask import Flask, request
from handler import sum

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello World!"

@app.route("/about")
def about():
    return "About Page"

@app.route("/cartoonify", methods=["POST"])
def image():
    input_iamge = request.files.get("image")
    

if __name__ == "__main__":
    app.run()

print(sum(2,3))


# how to access data coming from request body of axios method in flask python