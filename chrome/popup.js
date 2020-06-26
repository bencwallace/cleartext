let URL = "http://ec2-3-215-190-150.compute-1.amazonaws.com:5000/";
//let URL = "http://localhost:5000/";

var xhttp = new XMLHttpRequest();
// update popup text with server resonse
xhttp.onreadystatechange = function() {
	if (this.readyState === 4 && this.status === 200) {
		document.getElementById("text").innerHTML = this.responseText;
	}
};
// send request to server with text from background script
xhttp.open("POST", URL, true);
xhttp.setRequestHeader("Content-type", "application/json");
let selection = chrome.extension.getBackgroundPage().selection;
xhttp.send(JSON.stringify({
    "text": selection
}));
