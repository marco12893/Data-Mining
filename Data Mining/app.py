from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    print("Flask app running...")
    return render_template('index.html', title="Home", header="Welcome to Flask")

@app.route('/about')
def about():
    print("Flask app running...")
    return render_template('base.html', title="About", header="About Flask")

if __name__ == '__main__':
    app.run(debug=True)
    # skibidis
