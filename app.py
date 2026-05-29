from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():

    return jsonify({
        "score":99,
        "result":"NEW CODE WORKS"
    })