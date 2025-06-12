chrome.action.onClicked.addListener((tab) => {
    chrome.tabs.create({ url: "https://www.tradingview.com/chart/" }, (newTab) => {
        // Wait a moment for the page to load, then inject content.js
        setTimeout(() => {
            chrome.scripting.executeScript({
                target: { tabId: newTab.id },
                files: ["content.js"]
            });
        }, 5000); // wait 5 seconds
    });
});
