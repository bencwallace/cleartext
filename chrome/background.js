chrome.runtime.onMessage.addListener(receiver);

function receiver(request, sender, sendResponder) {
	window.selection = request.text;
}
