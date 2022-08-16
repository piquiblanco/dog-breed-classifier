import os

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from model import *

# file upload pattern sourced from https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/
UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app = Flask(__name__)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    # initial method – uploading file from the local machine to the flask app folder
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("prediction", name=filename))
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


from flask import send_from_directory


@app.route("/predictions/<name>")
def prediction(name):
    # human/dog detection and breed prediction using dog_breed_predict function
    img_file_path = "/static/" + name
    detection, breed = dog_breed_predict(img_file_path)
    if detection == "dog":
        message = "Dog detected in picture."
        breed = f"The predicted breed is {breed}."
    elif detection == "human":
        message = "Human detected in picture."
        breed = f"The predicted breed is {breed}."
    else:
        message = "No dog nor human detected in picture."
        breed = "No predicted breed."
    return render_template(
        "index.html", path=img_file_path, filename=name, message=message, breed=breed
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            # predict function
            prediction = dog_breed_predict(img_file_path)
            # delete uploaded file
            os.remove(img_file_path)
            return jsonify(prediction)
        # return True

    return True


if __name__ == "__main__":
    app.run()
