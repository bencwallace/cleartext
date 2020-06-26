// listen for text selections and send selected text to background script
window.addEventListener("mouseup", function() {
	let selectedText = window.getSelection().toString();
	if (selectedText.length > 0) {
		chrome.runtime.sendMessage({
			text: selectedText
		});
	}
});
