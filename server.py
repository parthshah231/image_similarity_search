# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import shutil
import os

# creating a Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "data/test"
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024  # 1 GB

# on the terminal type: python server.py
@app.route('/', methods = ['GET', 'POST'])


@app.route('/upload/', methods = ['POST'])
def upload():  
    if(request.method == 'POST'):    
      if(os.path.isdir('data/test')) :
        shutil.rmtree(app.config['UPLOAD_FOLDER']) # delete target folder
      os.makedirs("data/test")

      f = request.files['input']
      extension = os.path.splitext(f.filename)[1]
      f.filename = "input"+extension
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))) # save input image
      files = request.files.getlist("dataset[]")
      for file in files:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # save dataset images      
   
    return 'file uploaded successfully' # return json

# driver function
if __name__ == '__main__':
  
    app.run(debug = True)