from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os

app = Flask(__name__)
app.config.from_object('config')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/my-link')
def my_link():
    print("I got clicked!")
    return 'Click.'

@app.route('/uploader1', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
    #   f.save(secure_filename(f.filename))
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
    #   return f.filename+' uploaded successfully'
      return filter_one(f.filename)

def filter_one(filename):
    if filename != '':
        print("filename:", filename)
    return filename

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 9000)))