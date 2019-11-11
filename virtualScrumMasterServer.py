# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:42:29 2019

@author: Ishmeet
"""

from flask import Flask, jsonify, abort, request, make_response
import pickle
from scipy.sparse import hstack

app = Flask(__name__, static_url_path = "")

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found2(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

@app.route('/healthcheck', methods = ['GET'])
def healthcheck():
    healthcheck = {
        'healthcheck': "OK"
    }
    return jsonify( healthcheck ), 200

@app.route('/assignee', methods = ['POST'])
def predict_assignee():
    print("request.json:",request.json)
    if not request.json or not 'Summary' in request.json:
        abort(400)

    # Loading the saved model pickle
    Vector1 = pickle.load(open("Vector1.pkl", "rb"))
    Vector2 = pickle.load(open("Vector2.pkl", "rb"))
    Vector3 = pickle.load(open("Vector3.pkl", "rb"))
    Vector4 = pickle.load(open("Vector4.pkl", "rb"))
    Vector5 = pickle.load(open("Vector5.pkl", "rb"))
    Vector6 = pickle.load(open("Vector6.pkl", "rb"))
    tfidf_train = pickle.load(open("tfidf_train.pkl", "rb"))
    clf = pickle.load(open("MultinomialNB.pkl", "rb"))
    sgd = pickle.load(open("SGDClassifier.pkl", "rb"))
    Dtreeclf = pickle.load(open("DecisionTreeClassifier.pkl", "rb"))
    randomclf = pickle.load(open("RandomForestClassifier.pkl", "rb"))
    neigh = pickle.load(open("KNeighborsClassifier.pkl", "rb"))
    my_model = pickle.load(open("GridSearchCV.pkl", "rb"))

    print("Summary: ", request.json['Summary'])
    print ("Description: ", request.json['Description'])
    
    #Predict
    #abc = "FVLI_AM2:Event name should be modified for AIX_DOWN events"
    #cba = "==============================================START========================================== Environment details: Cluster type: Standard Number of nodes: 2 Nodes details with credentials: r1r2m1p79,r1r2m1p80,r1r2m1p85,r1r2m1p86 HMC details (If nodes"
    vector_test_1 = Vector1.transform([request.json['Summary']])
    vector_test_2 = Vector2.transform([request.json['Description']])
    vector_test_3 = Vector3.transform([""])
    vector_test_4 = Vector4.transform([""]) 
    vector_test_5 = Vector5.transform([""]) 
    vector_test_6 = Vector6.transform([""])
    vector_test_vector = hstack([vector_test_1, vector_test_2, vector_test_3, vector_test_4, vector_test_5, vector_test_6])
    vector_test_vector_tfidf = tfidf_train.transform(vector_test_vector)
    
    Dtreeclf_predicted = Dtreeclf.predict(vector_test_vector_tfidf)
    print ("Dtreeclf_predicted: ", Dtreeclf_predicted)
    MultinomialNB_predicted = clf.predict(vector_test_vector_tfidf)
    print ("MultinomialNB_predicted: ", MultinomialNB_predicted)
    SGDClassifier_predicted = sgd.predict(vector_test_vector_tfidf)
    print ("SGDClassifier_predicted: ", SGDClassifier_predicted)
    RandomForestClassifier_predicted = randomclf.predict(vector_test_vector_tfidf)
    print ("RandomForestClassifier_predicted: ", RandomForestClassifier_predicted)
    KNeighborsClassifier_predicted = neigh.predict(vector_test_vector_tfidf)
    print ("KNeighborsClassifier_predicted: ", KNeighborsClassifier_predicted)
    GridSearchCV_predicted = my_model.predict(vector_test_vector_tfidf)
    print ("GridSearchCV_predicted: ", GridSearchCV_predicted)
    
    assignee = {
        'assignee': ''.join(map(str,Dtreeclf_predicted))
    }

    return jsonify( assignee ), 200

@app.route('/rca', methods = ['POST'])
def predict_rca():
    print("request.json:",request.json)
    if not request.json or not 'Summary' in request.json:
        abort(400)

    # Loading the saved model pickle
    Vector1_2 = pickle.load(open("Vector1_2.pkl", "rb"))
    Vector2_2 = pickle.load(open("Vector2_2.pkl", "rb"))
    Vector3_2 = pickle.load(open("Vector3_2.pkl", "rb"))
    Vector4_2 = pickle.load(open("Vector4_2.pkl", "rb"))
    Vector5_2 = pickle.load(open("Vector5_2.pkl", "rb"))
    Vector6_2 = pickle.load(open("Vector6_2.pkl", "rb"))
    tfidf_train2 = pickle.load(open("tfidf_train2.pkl", "rb"))
    #clf2 = pickle.load(open("MultinomialNB2.pkl", "rb"))
    sgd2 = pickle.load(open("SGDClassifier2.pkl", "rb"))
    #Dtreeclf2 = pickle.load(open("DecisionTreeClassifier2.pkl", "rb"))
    #randomclf2 = pickle.load(open("RandomForestClassifier2.pkl", "rb"))
    #neigh2 = pickle.load(open("KNeighborsClassifier2.pkl", "rb"))
    
    X_test_2_1 = Vector1_2.transform([request.json['Summary']])
    X_test_2_2 = Vector2_2.transform([request.json['Description']])
    X_test_2_3 = Vector3_2.transform([""])
    X_test_2_4 = Vector4_2.transform([""])
    X_test_2_5 = Vector5_2.transform([""])
    X_test_2_6 = Vector6_2.transform([""])
    X_test2_vector = hstack([X_test_2_1, X_test_2_2, X_test_2_3, X_test_2_4, X_test_2_5, X_test_2_6])
    X_test_vector_tfidf2 = tfidf_train2.transform(X_test2_vector)

    predicted_sgd2 = sgd2.predict(X_test_vector_tfidf2)
    print ("predicted_sgd2: ", predicted_sgd2)
    rca = {
        'RCA': ''.join(map(str,predicted_sgd2))
    }

    return jsonify( rca ), 200
    
@app.route('/precision', methods = ['GET'])
def precision():
    precision= {
        'GridSearchCV': "84.53708191953466",
        'DecisionTreeClassifier': "84.00387784779447",
        'SGDClassifier': "81.19243819680078",
        'KNeighborsClassifier': "81.7741153659719",
        'MultinomialNB': "67.23218613669414",
        'RandomForestClassifier': "66.06883179835191"
    }
    return jsonify( precision ), 200
    
if __name__ == '__main__':
    app.run(debug=False)