from flask import Flask, render_template, send_file
from google_drive_request import simple_upload

UPLOAD_FOLDER = r'C:\Users\flynn\Desktop\figs'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
PICTURE = r'C:\Users\flynn\Desktop\figs\heatmap.png'


@app.route('/')
def index():
    return '''<!DOCTYPE html>
    <html lang="">
      <head>
        <title>Drive API Quickstart</title>
        <meta charset="utf-8" />
      </head>
      <body>
        <p>Graph Viewer</p>
        <img src="">
        <pre id="content" style="white-space: pre-wrap;"></pre>
        
      </body>
    </html>'''


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
