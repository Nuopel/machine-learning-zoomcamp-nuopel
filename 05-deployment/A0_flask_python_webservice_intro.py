from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    print("👉 /ping called")
    return 'pong', 200

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=9696)
