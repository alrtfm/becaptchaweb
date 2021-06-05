#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def clean_database(file):
    ''' Lee datos del .json exportado por Google Firebase y elimina los ID que 
    almacena Google por cada dato guardado.'''
    
    import json
    
    #Load database
    with open(file) as json_file:
        db = json.load(json_file)
       
    # -- Clean dictionary
    clean_db = {}
    
    for connection in db:
        
        #Almacena datos de cada conexion
        data_parsed = {
        'Device' : {},
        'Device Orientation' : {},
        'Google RECAPTCHA v3' : {},
        'Keyboard' : {},
        'Mouse' : {},
        'Accelerometer' : {},
        'Gyroscope' : {},
        #'Magnetometer' : {},
        'Touchscreen' : {},
        'SwipeGesture' : {}
        }
        
        # -- DEVICE DATA
        browsers_l = []
        isMobile_l = []
        userAgents_l = []
        # -- DEVICE ORIENTATION
        dev_ori_l = []
        dev_ori_a = []
        dev_ori_b = []
        dev_ori_g = []
        dev_ori_t = []
        # -- RECAPTCHA SCORE
        recap_score = []
        recap_success = []
        # -- KEYBOARD
        key_event = []
        key_keyname = []
        key_t = []
        # -- MOUSE
        mouse_event = []
        mouse_X = []
        mouse_Y = []
        mouse_t = []
        # -- ACCELEROMETER
        acc_type = []
        acc_X = []
        acc_Y = []
        acc_Z = []
        acc_t = []
        # -- GYROSCOPE
        gyro_type = []
        gyro_X = []
        gyro_Y = []
        gyro_Z = []
        gyro_t = []
        # -- TOUCHSCREEN
        touch_event = []
        touch_ID_abs = []
        touch_ID_rel = []
        touch_X = []
        touch_Y = []
        touch_t = []
        # -- SWIPE GESTURE
        swipe_ID_abs = []
        swipe_ID_rel = []
        swipe_X = []
        swipe_Y = []
        swipe_t = []
        
        
    # -- DEVICE
        try:
            device = db[connection]['Device']
            for key in device:
                browsers_l.append(device[key]['Browser'])
                isMobile_l.append(device[key]['IsMobile'])
                userAgents_l.append(device[key]['UserAgent'])
            data_parsed['Device']['Browser'] = browsers_l
            data_parsed['Device']['IsMobile'] = isMobile_l
            data_parsed['Device']['UserAgent'] = userAgents_l
        except:
            pass
    # -- DEVICE ORIENTATION  
        try:
            dev_ori = db[connection]['Device_Orientation']
            for key in dev_ori:
                dev_ori_l.append(dev_ori[key]['Absolute'])
                dev_ori_a.append(dev_ori[key]['alpha'])
                dev_ori_b.append(dev_ori[key]['beta'])
                dev_ori_g.append(dev_ori[key]['gamma'])
                dev_ori_t.append(dev_ori[key]['timestamp'])
            data_parsed['Device Orientation']['Absolute'] = dev_ori_l
            data_parsed['Device Orientation']['Alpha'] = dev_ori_a
            data_parsed['Device Orientation']['Beta'] = dev_ori_b
            data_parsed['Device Orientation']['Gamma'] = dev_ori_g
            data_parsed['Device Orientation']['Timestamp'] = dev_ori_t
                
        except:
            pass
    # -- RECAPTCHA SCORE
        try:
            recap = db[connection]['Google RECAPTCHA v3']
            for key in recap:
                recap_score.append(recap[key]['Score'])
                recap_success.append(recap[key]['Success'])
            data_parsed['Google RECAPTCHA v3']['Score'] = recap_score
            data_parsed['Google RECAPTCHA v3']['Success'] = recap_success
                
        except:
            pass
    # -- KEYBOARD
        try:
            kb = db[connection]['Keyboard']
            for key in kb:
                key_event.append(kb[key]['event'])
                key_keyname.append(kb[key]['keyname'])
                key_t.append(kb[key]['timestamp'])
            data_parsed['Keyboard']['Event'] = key_event
            data_parsed['Keyboard']['Keyname'] = key_keyname
            data_parsed['Keyboard']['Timestamp'] = key_t
        except:
            pass
    # -- MOUSE
        try:
            mouse = db[connection]['Mouse']
            for key in mouse:
                mouse_event.append(mouse[key]['event'])
                mouse_X.append(mouse[key]['X'])
                mouse_Y.append(mouse[key]['Y'])
                mouse_t.append(mouse[key]['timestamp'])
            data_parsed['Mouse']['Event'] = mouse_event
            data_parsed['Mouse']['X'] = mouse_X
            data_parsed['Mouse']['Y'] = mouse_Y
            data_parsed['Mouse']['Timestamp'] = mouse_t
                
        except:
            pass
    # -- ACCELEROMETER
        try:
            acc = db[connection]['Accelerometer']
            for key in acc:
                acc_type.append(acc[key]['type'])
                acc_X.append(acc[key]['X'])
                acc_Y.append(acc[key]['Y'])
                acc_Z.append(acc[key]['Z'])
                acc_t.append(acc[key]['timestamp'])
            data_parsed['Accelerometer']['Type'] = acc_type
            data_parsed['Accelerometer']['X'] = acc_X
            data_parsed['Accelerometer']['Y'] = acc_Y
            data_parsed['Accelerometer']['Z'] = acc_Z
            data_parsed['Accelerometer']['Timestamp'] = acc_Z
                
        except:
            pass
        try:
    # -- GYROSCOPE
            gyro = db[connection]['Gyroscope']
            for key in gyro:
                gyro_type.append(gyro[key]['type'])
                gyro_X.append(gyro[key]['X'])
                gyro_Y.append(gyro[key]['Y'])
                gyro_Z.append(gyro[key]['Z'])
                gyro_t.append(gyro[key]['timestamp'])
            data_parsed['Gyroscope']['Type'] = gyro_type
            data_parsed['Gyroscope']['X'] = gyro_X
            data_parsed['Gyroscope']['Y'] = gyro_Y
            data_parsed['Gyroscope']['Z'] = gyro_Z
            data_parsed['Gyroscope']['Timestamp'] = gyro_t
        except:
            pass
        try:
    # -- TOUCHSCREEN
            touch = db[connection]['Touchscreen']
            for key in touch:
                touch_event.append(touch[key]['event'])
                touch_ID_abs.append(touch[key]['touchID_abs'])
                touch_ID_rel.append(touch[key]['touchID_rel'])
                touch_X.append(touch[key]['X'])
                touch_Y.append(touch[key]['Y'])
                touch_t.append(touch[key]['timestamp'])
            data_parsed['Touchscreen']['Event'] = touch_event
            data_parsed['Touchscreen']['ID_abs'] = touch_ID_abs
            data_parsed['Touchscreen']['ID_rel'] = touch_ID_rel
            data_parsed['Touchscreen']['X'] = touch_X
            data_parsed['Touchscreen']['Y'] = touch_Y
            data_parsed['Touchscreen']['Timestamp'] = touch_t
        except:
            pass
        try:
    # -- SWIPE GESTURE
            swipe = db[connection]['SwipeGesture']
            for key in swipe:
                swipe_ID_abs.append(swipe[key]['swipeID_abs'])
                swipe_ID_rel.append(swipe[key]['swipeID_rel'])
                tempXvector = []
                tempYvector = []
                for sample in swipe[key]['X']:
                    tempXvector.append(sample)
                swipe_X.append(tempXvector)
                for sample in swipe[key]['Y']:
                    tempYvector.append(sample)
                swipe_Y.append(tempYvector)
                swipe_t.append(swipe[key]['timestamp'])
                
            data_parsed['SwipeGesture']['ID_abs'] = swipe_ID_abs
            data_parsed['SwipeGesture']['ID_rel'] = swipe_ID_rel
            data_parsed['SwipeGesture']['X'] = swipe_X
            data_parsed['SwipeGesture']['Y'] = swipe_Y
            data_parsed['SwipeGesture']['Timestamp'] = swipe_t
        except Exception as e:
            print(e)
            pass
        
        #Guarda datos de la conexion
        clean_db[connection] = data_parsed
        
    return clean_db
