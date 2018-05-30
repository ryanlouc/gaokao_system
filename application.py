#!/usr/bin/env python
# -*- coding:UTF-8-*-


from flask import Flask, render_template, request, session, send_file, abort
from flask_bootstrap import Bootstrap


from datetime import datetime

import os
import sys
import json
import codecs
import numpy as np


ALLOWED_EXTENSIONS = set(["txt"])

UPLOAD_PATH = "./data"
GOLD_PATH = os.path.join(UPLOAD_PATH, "gold/test_y.txt")
DUMP_PATH = os.path.join(UPLOAD_PATH, "dumps")
SUBMIT_PATH = os.path.join(UPLOAD_PATH, "submit")
DUMP_FILE = os.path.join(UPLOAD_PATH, "dump")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_PATH
app.config["SECRET_KEY"] = "hahah blablab"
bootstrap = Bootstrap(app)


@app.route("/")
def overview():
    return render_template("overview.html")


def convert2int(string):
    return int(datetime.strptime(string, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S"))


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


@app.route("/submission.html", methods=["GET", "POST"])
def submission():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], "test.txt")
            file.save(filepath)

            session["count"] = 0
            os.system("sh killprocess.sh 'nohup python solve_problem.py &'")

    return render_template("submission.html")


@app.route("/solve.html")
def solve():
    solve_result = ["waiting for problem-solving log....".encode("utf-8")]

    if "count" in session.keys():
        session["count"] += 1
        if session["count"] == 1:
            os.system("nohup python solve_problem.py &")

    if os.path.exists(os.path.abspath("data/test.info")):
        with codecs.open("data/test.info", "r", "utf-8") as f:
            solve_result = f.readlines()

    return render_template("solve.html", solve_result=solve_result)


def extract_problem(sentence):
    return [("".join(line.split(" "))) for line in sentence]


def extract_ana(sentence):
    result = []
    for line in sentence:
        temp = line.split(" ")
        result.append("".join(temp[:-1])+" "+temp[-1])
    return result


def extract_choice(sentence):
    line = sentence
    temp_line = np.array([float(score) for score in line.split("\t")])
    index = np.argmax(temp_line)

    final_result = ""
    if index == 0:
        final_result = "A"
    elif index == 1:
        final_result = "B"
    elif index == 2:
        final_result = "C"
    elif index == 3:
        final_result = "D"
    return line, final_result


@app.route("/analysis.html")
def analysis():
    final_result = [[["background" , "query", "choice A", "choice B", "choice C", "choice D"],
                     ["background_ner" , "query_ner", "choice A_ner", "choice B_ner", "choice C_ner", "choice D_ner"],
                     ["score", "score", "score", "score", "score", "score", "score", "score",], "four score", "final choice"]]
    if os.path.exists(os.path.abspath("data/test_probs.txt")):

        with codecs.open("data/test_probs.txt", "r", "utf-8") as f:
            probs_result = [line.strip() for line in f.readlines()]

        with codecs.open("data/test_prob_ner.txt", "r", "utf-8") as f:
            ner_result = [line.strip() for line in f.readlines()]

        with codecs.open("data/test_ana_score.txt", "r", "utf-8") as f:
            text_ana_score = [line.strip() for line in f.readlines()]

        final_result = []
        problem_number = len(probs_result)
        for problem_id in range(problem_number):
            problem = extract_problem(text_ana_score[problem_id * 14: problem_id * 14 + 6])
            ner = ner_result[problem_id*6: problem_id*6+6]
            ana = extract_ana(text_ana_score[problem_id * 14 + 6: problem_id * 14 + 14])
            score, final_choice = extract_choice(probs_result[problem_id])

            one_result = []
            one_result.append(problem)
            one_result.append(ner)
            one_result.append(ana)
            one_result.append(score)
            one_result.append(final_choice)

            final_result.append(one_result)

        with codecs.open("data/test_final_result.txt", "w", "utf-8") as f:
            json.dump(final_result, f)

    return render_template("analysis.html", final_result=final_result)


@app.route("/download.html")
def download():
    if request.method=="GET":
        if os.path.exists(os.path.abspath("data/test_final_result.json")):
            return send_file("data/test_final_result.json", as_attachment=True)
        abort(404)


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500


if __name__ == "__main__":
    ip = "114.212.190.232"
    port = 33333
    if len(sys.argv) == 3:
        ip, port = str(sys.argv[1]), int(sys.argv[2])
    app.run(host=ip, port=port)
    # app.run(host="127.0.0.1", port=33333)


