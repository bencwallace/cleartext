// listen for message from content script and update selection accordingly
chrome.runtime.onMessage.addListener(function(request, sender, sendResponder) {
    console.log('received request');
	window.selection = request.text;
});
