// listen for message from content script and update selection accordingly
chrome.runtime.onMessage.addListener(function(request, sender, sendResponder) {
	window.selection = request.text;
});
