#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def download_db(filename):
    import pyrebase
    import json
    
    config = {
        "apiKey": "AIzaSyCJsZvx1OkMhpd2Mz51W8PkylsFRQSUZiU",
        "authDomain": "becaptchaweb.firebaseapp.com",
        "databaseURL": "https://becaptchaweb.firebaseio.com/",
        "storageBucket": "becaptchaweb.appspot.com"
        }
      
    # initialisatiing pyrebase
    firebase = pyrebase.initialize_app(config)
    
    # initialisatiing Database
    db = firebase.database()
    
    data = {}
    #Get all DB
    #sesions = db.get().pyres
    
    #Get latest connection only
    sesions = db.order_by_key().limit_to_last(1).get().pyres
    
    for sesion in sesions:
        data[sesion.key()] = sesion.val()
    
    with open(filename,"w") as fp:
        json.dump(data, fp, indent=4)
