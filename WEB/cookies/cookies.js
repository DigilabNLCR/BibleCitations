// Cookie management
let cookiesInfo = document.getElementById('cookies_info');
let cookiesButton = document.getElementById('cookies_info_button');
let cookiesDeclineButton = document.getElementById('decline');

function cookiesInfoDisAppear() {
    cookiesInfo.style.transform = "translateY(500px)";
};

function cookiesInfoAppear() {
    cookiesInfo.style.transform = "translateY(0px)";
};

cookiesButton.onclick = function() {cookiesInfoAppear()};

cookiesDeclineButton.onclick = function() {cookiesInfoDisAppear()};