chrome.runtime.onMessage.addListener(receiver);

function receiver(request, sender, sendResponder) {
    console.log('received request')
	window.selection = request.text;
}
