from flask import Flask, render_template, request, redirect

app = Flask(__name__)

app.vars={}


@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')



# Custom functions:

@app.route('/gettext', methods=['POST'])
def gettext():
    if request.method == 'POST':
        #text = request.form['text']
        #f = open('text.txt', 'w')
        #f.write(text)
        #f.close()
	return request.form['text']
    return 1





if __name__ == '__main__':
  app.run(port=33507)
