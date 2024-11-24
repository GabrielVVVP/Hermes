from flask import Flask, request, jsonify
from chatllm import chat_response

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Please provide a message!"}), 400
    
    response = chat_response(user_input)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug="True")