// ALL FUNCTIONS MADE TO CAPTURE SENSORS

var userid = getUserID();
window.freq = 60; // 200 - Capturas por segundo de los sensores
window.totalTouches = 0; //Registrar TOTAL de los toques del Touchpad

var ongoingTouches = [];
var initialtime; //Registro del tiempo en UNIX time en milisegundos

//FIREBASE -----------------------------------------------------------------------

function loadFirebase(){
	var firebaseConfig = {
	apiKey: "AIzaSyCJsZvx1OkMhpd2Mz51W8PkylsFRQSUZiU",
 	authDomain: "becaptchaweb.firebaseapp.com",
 	databaseURL: "https://becaptchaweb.firebaseio.com",
 	projectId: "becaptchaweb",
 	storageBucket: "becaptchaweb.appspot.com",
 	messagingSenderId: "971421400464",
 	appId: "1:971421400464:web:a83b8704dd4c7e32b0c715"
	};

	firebase.initializeApp(firebaseConfig);
}

function pushToFirebase(userid,child,data){
	//Meter datos (data) en el child especificado
	//userid identifica la conexion. 
	//Child identifica el tipo de datos/sensor: Mouse, Keyboard...
	firebase.database().ref('/'+userid+'/'+child).push(data)
}

//Funciones generales  -----------------------------------------------------------------------

function getUserID(){
	//Asigna un ID a la conexion a la web
	var timestamp = new Date().getTime();
	return 'connection_'+timestamp;
}

function stopSensors(){
	//Para la captura de datos de los sensores del movil
	console.log("Stopping sensor capture...");
	window.removeEventListener("deviceorientation", handleOrientation, true);
	try {window.accelerometer.stop();} catch(err) {console.log(err);};
	try {window.linear_accelerometer.stop();} catch(err) {console.log(err);};
	try {window.magnetometer.stop();} catch(err) {console.log(err);};	
	try {window.gyroscope.stop();} catch(err) {console.log(err);};
}

//DEVICE INFO -----------------------------------------------------------------------

function getDeviceInfo(){
    console.log('Detected browser: '+getBrowser());
    console.log('Detected User Agent: '+navigator.userAgent);
    pushToFirebase(userid,'Device',{"IsMobile": isMobile(),"UserAgent": navigator.userAgent, "Browser": getBrowser()});
}

function isMobile(){
	//Detecta si es un movil
	//Source: https://stackoverflow.com/questions/58153528/how-to-detect-a-mobile-device
	let hasTouchScreen = false;
	if ("maxTouchPoints" in navigator) {
		hasTouchScreen = navigator.maxTouchPoints > 0;
	} else if ("msMaxTouchPoints" in navigator) {
		hasTouchScreen = navigator.msMaxTouchPoints > 0;
	} else {
		let mQ = window.matchMedia && matchMedia("(pointer:coarse)");
		if (mQ && mQ.media === "(pointer:coarse)") {
			hasTouchScreen = !!mQ.matches;
		} else if ('orientation' in window) {
			hasTouchScreen = true;
		} else {
			let UA = navigator.userAgent;
			hasTouchScreen = (
				/\b(BlackBerry|webOS|iPhone|IEMobile)\b/i.test(UA) ||
				/\b(Android|Windows Phone|iPad|iPod)\b/i.test(UA)
				);
		}
	}
	let mQ2 = window.matchMedia && matchMedia("(max-width: 767px), (max-height: 767px)");
	return ((hasTouchScreen === true) && (mQ2.matches === true));
}

//BROWSER INFO -----------------------------------------------------------------------

function getBrowser(){
	//Detecta el navegador via Duck-Typing. Si no funciona, utiliza el metodo de leer el User Agent, aunque es menos recomendado.
	var browser = getBrowserDuckTyping()
	if (browser=='unknown'){
		return getBrowserUserAgent();
	} else{
		return browser;
	}
}
function getBrowserDuckTyping(){
	//Detecta el navegador via Duck-Typing.
	var browser = 'unknown';
	if (!!window.chrome && (!!window.chrome.webstore || !!window.chrome.runtime)) {
		// Chrome version 1 to 71
		browser = 'Chrome';
	} else if (typeof InstallTrigger !== 'undefined'){
		// Firefox version 1.0+
		browser = 'Firefox';
	} else if ((!!window.opr && !!opr.addons) || !!window.opera || navigator.userAgent.indexOf(' OPR/') >= 0){
		// Opera version 8.0+
		browser = 'Opera';
	} else if (/constructor/i.test(window.HTMLElement) || (function (p) { return p.toString() === "[object SafariRemoteNotification]"; })(!window['safari'] || (typeof safari !== 'undefined' && safari.pushNotification))){
		// Safari version 3.0+ "[object HTMLElementConstructor]" 
		browser = 'Safari';
	} else if (/*@cc_on!@*/false || !!document.documentMode){
		// Internet Explorer version 6-11
		browser = 'Internet Explorer';
	} else if (!(/*@cc_on!@*/false || !!document.documentMode) && !!window.StyleMedia){
		// Edge version 20+
		browser = 'Edge';
	} 
	if ((!!window.chrome && (!!window.chrome.webstore || !!window.chrome.runtime)) && (navigator.userAgent.indexOf("Edg") != -1)){
		// Edge detection (Based on chromium)
		browser = 'Edge';
	}
	return browser;
}
function getBrowserUserAgent(){
	//Detecta el navegador via User Agent
	var browser = 'unknown';
	var ua = navigator.userAgent;
	if ((ua.indexOf("Opera") || ua.indexOf('OPR')) != -1) {
		browser = 'Opera'
	} else if (ua.indexOf("Edge") != -1) {
		browser = 'Edge'
	} else if (ua.indexOf("Chrome") != -1) {
		browser = 'Chrome'
	} else if (ua.indexOf("Safari") != -1) {
		browser = 'Safari'
	} else if (ua.indexOf("Firefox") != -1) {
		browser = 'Firefox'
	} else if ((ua.indexOf("MSIE") != -1) || (!!document.documentMode == true)) {
		browser = 'Internet Explorer'
	} else {
		browser = 'unknown'
	}
	return browser;
}

//MOUSE -----------------------------------------------------------------------

function mouseMove(event){
	let x=event.clientX;
	let y=event.clientY;
	var mstime = new Date().getTime();
	console.log("Mouse position (x,y) =  (" + x +","+ y+") at = "+mstime+" ms");
	//Subirlo a Firebase	
	pushToFirebase(userid,'Mouse',{"event": "Move", "timestamp": mstime, "X": x,"Y": y,});
}
function mouseDown(event) { 
	let x = event.clientX; 
	let y = event.clientY; 
	var mstime = new Date().getTime();
	console.log("Latest click on: (x,y) =  (" + x +","+ y+") at = "+mstime+" ms");
	//Subirlo a Firebase	
	pushToFirebase(userid,'Mouse',{"event": "Click", "timestamp": mstime,  "X": x,"Y": y,});
}
function mouseLeave(event){
	x=event.clientX;
	y=event.clientY;
	console.log("Mouse left window on (x,y) =  (" + x +","+ y+")")
}

//TOUCHSCREEN -----------------------------------------------------------------------

function touchStart(event) {  
    // Registrar toques y mostrar coordenadas iniciales y numero de toques simultaneos
	window.totalTouches = window.totalTouches +1;
    console.log('********************************');
    console.log('NUEVO TOQUE');
    console.log('Numero de toque (absoluto) : ' + window.totalTouches);
    console.log('Numero de toques simultaneos : ' + event.touches.length);
    var mstime = new Date().getTime(); 
    initialtime = mstime;
    for (var i = 0; i < event.touches.length; i++) {
        ongoingTouches.push(copyTouch(event.touches[i])); //Registrar toque     
        var x = event.touches[i].pageX;
        var y = event.touches[i].pageY;
        console.log('(X,Y) = (' + x +','+ y +')');
        console.log('Posición del toque ' + i + ' en t = ' + mstime + 'ms =' );
        pushToFirebase(userid,'Touchscreen',{"event": "touchStart", "timestamp": mstime,"touchID_abs":window.totalTouches,"touchID_rel":i, "X": x,"Y": y,});
    }
}
function touchMove(event) {
    //Registrar los puntos por los que se desplaza cada toque en la pantalla
    //Loguear los movimientos del toque
    console.log('********************************');
    for (var i = 0; i < event.touches.length; i++) {
    	var mstime = new Date().getTime(); 
        //Identifica el primer toque que inicia el movimiento
        var idx = ongoingTouchIndexById(event.touches[i].identifier);
        if (idx >= 0) {
        	var x = event.touches[i].pageX;
        	var y = event.touches[i].pageY;
        	console.log('(X,Y) = (' + x +','+ y +')')
        	console.log('Desplazamiento del toque ' + i + ' en t = ' + mstime +' ms a: ');
        	pushToFirebase(userid,'Touchscreen',{"event": "touchMove", "timestamp": mstime,"touchID_abs":window.totalTouches,"touchID_rel":i,"X": x,"Y": y,});
        }
    }
}
function touchEnd(event) {  
    //Gestionar el fin de un toque
    console.log('********************************');
    console.log('FIN DEL TOQUE');
    var mstime = new Date().getTime();
    touchtime = mstime - initialtime;
    var touches = event.changedTouches;
    for (var i = 0; i < touches.length; i++) {
    	var idx = ongoingTouchIndexById(touches[i].identifier);
    	if (idx >= 0) {
    		var x = touches[i].pageX;
    		var y = touches[i].pageY;
    		console.log('(X,Y) = (' + x +','+ y +')');
    		console.log('Posición del último toque ' + i + ' en t = ' + mstime + 'ms = ');
    		console.log('Duración del toque ' + i + ' = ' + touchtime + 'ms');
    		pushToFirebase(userid,'Touchscreen',{"event": "touchEnd", "timestamp": mstime,"touchID_abs":window.totalTouches,"touchID_rel":idx,"X": x,"Y": y,});
            ongoingTouches.splice(idx, 1);  // Eliminar toque
        }
    }
}
function touchCancel(event) {
    //Gestionar toques cancelados
    event.preventDefault();            
    for (var i = 0; i < event.touches.length; i++) {
    	var mstime = new Date().getTime(); 
    	var idx = ongoingTouchIndexById(event.touches[i].identifier);
        ongoingTouches.splice(idx, 1);  // Eliminar toque
        console.log('Toque ' + i + ' cancelado en t = ' + mstime +'ms');
    }
}
// Identifican el toque actual para poder seguirlo
function copyTouch({ identifier, pageX, pageY }) {
	return { identifier, pageX, pageY };
}
function ongoingTouchIndexById(idToFind) {
	for (var i = 0; i < ongoingTouches.length; i++) {
		var id = ongoingTouches[i].identifier;
		if (id == idToFind) 
			{return i;}
            } return -1;    // not found
        }

//KEYBOARD -----------------------------------------------------------------------

function keyDown(event) { 
	const keyName = event.key;
	var mstime = new Date().getTime();
	if (keyName === 'Control') { // do not alert when only Control key is pressed.
		return;
	} 
	if (event.ctrlKey) {
		console.log('Combination of ctrlKey ' + keyName + ' at time = '+mstime+'ms');
		pushToFirebase(userid,'Keyboard',{"event": "keyDown", "timestamp": mstime,  "keyname":'Control + '+keyName,});
	} else {
		console.log('Key pressed: ' + keyName+ ' at time = '+mstime+'ms');
		pushToFirebase(userid,'Keyboard',{"event": "keyDown", "timestamp": mstime,  "keyname":keyName,});
	}
}
function keyUp(event) { 
	const keyName = event.key;
	var mstime = new Date().getTime();
	console.log(keyName+' key was released at time = '+mstime+'ms');
	pushToFirebase(userid,'Keyboard',{"event": "keyUp", "timestamp": mstime,  "keyname":keyName,});
}

// ASK FOR PERMISSION AND RUN -----------------------------------------------

function askPermissionAndRun(sensor,sensorfunction){
    navigator.permissions.query({ name: sensor }).then(result => {
        if (result.state === 'denied') {
            console.log('Permission to use '+sensor+'sensor is denied.')
            return;
        }else{
            console.log('Permission to use '+sensor+'sensor is granted.')
        }
    });
}

//ACCELEROMETER -----------------------------------------------------------------------

function accelerometer(){
    var mstime = new Date().getTime();
    if ( 'Accelerometer' in window ) {
        let sensor = new Accelerometer({frequency: window.freq});
        sensor.addEventListener('reading', function(e) {
            console.log('Accelerometer reads: x: ' + e.target.x + ' y: ' + e.target.y + '  z: ' + e.target.z);
            pushToFirebase(userid,'Accelerometer',{"type": "Absolute", "timestamp": mstime, "X": e.target.x,"Y": e.target.y,"Z": e.target.z,});
        });
        window.accelerometer = sensor;
        window.accelerometer.start();
    }else console.log('Accelerometer not supported');
    if ( 'LinearAccelerationSensor' in window ) {
        //Linear Acceleration
        let laSensor = new LinearAccelerationSensor({frequency: window.freq});
        laSensor.addEventListener('reading', e => {
            console.log('Linear accelerometer reads x: ' + laSensor.x+ ' y: ' + laSensor.y + ' z: ' + laSensor.z);
            pushToFirebase(userid,'Accelerometer',{"type": "Linear", "timestamp": mstime, "X": laSensor.x, "Y": laSensor.y, "Z": laSensor.z,});
        });
        window.linear_accelerometer = laSensor;
        window.linear_accelerometer.start();
    }else console.log('Linear accelerometer not supported');
}

//MAGNETOMETER -----------------------------------------------------------------------

function magnetometer(){
    var mstime = new Date().getTime();
    if ( 'Magnetometer' in window ) {
      let sensor = new Magnetometer({frequency: window.freq});
      sensor.addEventListener('reading', function(e) {
        console.log('Magnetometer reads x: ' + sensor.x + ' y: ' + sensor.y + ' z: ' + sensor.z);
        pushToFirebase(userid,'Magnetometer',{"timestamp": mstime, "X": sensor.x, "Y": sensor.y, "Z": sensor.z,});
    });
      window.magnetometer = sensor;
      window.magnetometer.start();
  }
  else console.log('Magnetometer not supported');
}

//GYROSCOPE -----------------------------------------------------------------------

function gyroscope(){
    var mstime = new Date().getTime();
    if ( 'Gyroscope' in window ) {
      let sensor = new Gyroscope({frequency: window.freq});
      sensor.addEventListener('reading', function(e) {
        console.log('Gyroscope reads x: ' + e.target.x + ' y: ' + e.target.y + '  z: ' + e.target.z);
        pushToFirebase(userid,'Gyroscope',{"timestamp": mstime, "X": e.target.x, "Y": e.target.y, "Z": e.target.z,});
    });
      window.gyroscope = sensor;
      window.gyroscope.start();
  } else console.log('Gyroscope not supported');
}

//DEVICE ORIENTATION -----------------------------------------------------------------------

function handleOrientation(event) {
    var mstime = new Date().getTime();
    console.log('Device Orientation reads absolute = '+ event.absolute+' alpha ='+ event.alpha+' beta ='+ event.beta+' gamma ='+ event.gamma);
    pushToFirebase(userid,'Device_Orientation',{"timestamp": mstime, "Absolute": event.absolute, "alpha": event.alpha, "beta": event.beta, "gamma": event.gamma,});
}

//-------------------------------------------------------------------------------------------
//------------ DETECCION DE INTERACION TACTIL Y ACELEROMETRO PARA EL DEMOSTRADOR ------------
//-------------------------------------------------------------------------------------------


var xDown = null;                                                        
var yDown = null;
window.RightSwipeDone = 0;
var xreads = []
var yreads = []
var tsreads = []
window.laccduringswipe_x = []
window.laccduringswipe_y = []
window.laccduringswipe_z = []
window.accelerometer_supported = sessionStorage.getItem("accelerometer_supported");


function detectSwipeRight(event) {
    if ( ! xDown || ! yDown ) {
        return;
    }
    var xUp = event.touches[0].clientX;                                    
    var yUp = event.touches[0].clientY;

    var xDiff = xDown - xUp;
    var yDiff = yDown - yUp;
    if(Math.abs( xDiff )+Math.abs( yDiff )>150){ //to deal with to short swipes

      if ( Math.abs( xDiff ) > Math.abs( yDiff ) ) {/*most significant*/
        if ( xDiff <= 0 ){
        	/* right swipe */
            console.log("right swipe") 
            window.RightSwipeDone = 1;
        }                       
	  }
      /* reset values */
    xDown = null;
    yDown = null;
    } 
}

function handleTouchStart(event) { 
  //Detectar swipe right
  xDown = event.touches[0].clientX;                                      
  yDown = event.touches[0].clientY;  
  // Registrar toques y mostrar coordenadas iniciales y numero de toques simultaneos
  sq.totalTouches = sq.totalTouches +1;
    var mstime = new Date().getTime(); 
    initialtime = mstime;
    for (var i = 0; i < event.touches.length; i++) {
        ongoingTouches.push(copyTouch(event.touches[i])); //Registrar toque     
        var x = event.touches[i].pageX;
        var y = event.touches[i].pageY;
        xreads.push(x)
        yreads.push(y)
        tsreads.push(mstime)
        console.log("swipe start")
        //Arrancar la captura del acelerometro
        if (window.accelerometer_supported != 0){
            window.laccduringswipe_x = []
            window.laccduringswipe_y = []
            window.laccduringswipe_z = []
            accelerometer_during_swipe()
        }
    }
}
function handleTouchMove(event) {
    //Registrar los puntos por los que se desplaza cada toque en la pantalla
    //Loguear los movimientos del toque
    for (var i = 0; i < event.touches.length; i++) {
      var mstime = new Date().getTime(); 
        //Identifica el primer toque que inicia el movimiento
        var idx = ongoingTouchIndexById(event.touches[i].identifier);
        if (idx >= 0) {
	        var x = event.touches[i].pageX;
	        var y = event.touches[i].pageY;
	        xreads.push(x)
	        yreads.push(y)
	        tsreads.push(mstime)
	        console.log("swipe move")
			//Obtencion de coordenadas de la caja de swipe relativas a la ventana
	        var rect = sq.getBoundingClientRect();
			console.log(rect.top, rect.right, rect.bottom, rect.left);
			var rr = rect.right;var rl = rect.left;var rb = rect.bottom;var rt = rect.top;

			//Si sale de la caja, subo lo que tenga y reseteo el toque
			if (x<rect.left || x>rect.right || y>rect.botom || y<rect.top){
				if(window.RightSwipeDone==1){

                    //Se para captura de acelerometro y se guardan los datos
                    if (window.accelerometer_supported == 1){
                        window.linear_accelerometer.stop();
                        sessionStorage.setItem("laccduringswipe_x", JSON.stringify(window.laccduringswipe_x));
                        sessionStorage.setItem("laccduringswipe_y", JSON.stringify(window.laccduringswipe_y));
                        sessionStorage.setItem("laccduringswipe_z", JSON.stringify(window.laccduringswipe_z));
                    }


					//Normalizar secuencia a esquina superior izquierda
					xreads2 = []; yreads2 = [];
					xreads.forEach((value, index) => {xreads2.push((xreads[index] - xreads[0])/(rr-rl))});
					yreads.forEach((value, index) => {yreads2.push((yreads[index] - yreads[0])/(rb-rt))});
					xreads2[0] = 0; yreads2[0] = 0;
	          		pushToFirebase(userid,'SwipeGesture',{"timestamp": tsreads,"swipeID_abs":window.totalTouches,"swipeID_rel":idx,"X": xreads2,"Y": yreads2,});
                    
                    //Enviarlo a FLASK
                    var seq = {x:xreads2,y:yreads2};
                    $.ajax({
                        type: 'POST',
                        url: "/rcv_swipe",
                        contentType: 'application/json;charset=UTF-8',  
                        data: JSON.stringify(seq),
                        success: function (data) {
                            //Añadir el codigo HTML recibido de FLASK a la web a traves del objeto HTML 'output'
                            $('#output').html(data);
                        }
                    });
                    console.log("Movimiento registrado")

				}
				// Eliminar toque y resetear arrays
		        ongoingTouches.splice(idx, 1);
		        xreads = []
		        yreads = []
		        window.RightSwipeDone = 0;
			}
		}
	}
}

function handleTouchEnd(event) {  
    //Gestionar el fin de un toque
    var mstime = new Date().getTime();
    touchtime = mstime - initialtime;
    var touches = event.changedTouches;
    for (var i = 0; i < touches.length; i++) {
      var idx = ongoingTouchIndexById(touches[i].identifier);
      if (idx >= 0) {
        var x = touches[i].pageX;
        var y = touches[i].pageY;
        xreads.push(x)
        yreads.push(y)
        tsreads.push(mstime)
        console.log("swipe end")
        //Se para captura de acelerometro y se guardan los datos
        if (window.accelerometer_supported == 1){
            window.linear_accelerometer.stop();
            sessionStorage.setItem("laccduringswipe_x", JSON.stringify(window.laccduringswipe_x));
            sessionStorage.setItem("laccduringswipe_y", JSON.stringify(window.laccduringswipe_y));
            sessionStorage.setItem("laccduringswipe_z", JSON.stringify(window.laccduringswipe_z));
            }

        // Si hubo swipe right, subir a firebase
        if(window.RightSwipeDone==1){
			var rect = sq.getBoundingClientRect();
			var rr = rect.right;var rl = rect.left;var rb = rect.bottom;var rt = rect.top;
			xreads2 = []; yreads2 = [];
			xreads.forEach((value, index) => {xreads2.push((xreads[index] - xreads[0])/(rr-rl))});
			yreads.forEach((value, index) => {yreads2.push(-((yreads[index] - yreads[0])/(rb-rt)))});
			xreads2[0] = 0; yreads2[0] = 0;
        	pushToFirebase(userid,'SwipeGesture',{"timestamp": tsreads,"swipeID_abs":window.totalTouches,"swipeID_rel":idx,"X": xreads2,"Y": yreads2,});

            //Enviarlo a FLASK
            var seq = {x:xreads2,y:yreads2};
            $.ajax({
                type: 'POST',
                url: "/rcv_swipe",
                contentType: 'application/json;charset=UTF-8',  
                data: JSON.stringify(seq),
                success: function (data) {
                    //Añadir el codigo HTML recibido de FLASK a la web a traves del objeto HTML 'output'
                    $('#output').html(data);
                }
            });
            console.log("Movimiento registrado")
        }
        // Eliminar toque y resetear arrays
        ongoingTouches.splice(idx, 1);
        xreads = [];
        yreads = [];
        window.RightSwipeDone = 0;
        }
    }
}
function handleTouchCancel(event) {
    //Gestionar toques cancelados
    event.preventDefault();            
    for (var i = 0; i < event.touches.length; i++) {
      var mstime = new Date().getTime(); 
      var idx = ongoingTouchIndexById(event.touches[i].identifier);
        // Eliminar toque y resetear arrays
        ongoingTouches.splice(idx, 1);
        xreads = []
        yreads = []
        window.RightSwipeDone = 0;
    }
}


function accelerometer_during_swipe(){
    var mstime = new Date().getTime();
    //Linear Acceleration
    if ( 'LinearAccelerationSensor' in window ) {
        let laSensor = new LinearAccelerationSensor({frequency: window.freq});
        laSensor.addEventListener('reading', function(e) {
            console.log('Linear accelerometer reads x: ' + laSensor.x+ ' y: ' + laSensor.y + ' z: ' + laSensor.z);
            window.laccduringswipe_x.push(laSensor.x)
            window.laccduringswipe_y.push(laSensor.y)
            window.laccduringswipe_z.push(laSensor.z)
            pushToFirebase(userid,'Accelerometer',{"type": "Linear", "timestamp": mstime, "X": laSensor.x, "Y": laSensor.y, "Z": laSensor.z,});
        });
        window.linear_accelerometer = laSensor;
        window.linear_accelerometer.start();
        sessionStorage.setItem("accelerometer_supported", JSON.stringify(1));
    }else {
        sessionStorage.setItem("accelerometer_supported", JSON.stringify(0));
        alert('No puede leerse el sensor de aceleración');
    }
}