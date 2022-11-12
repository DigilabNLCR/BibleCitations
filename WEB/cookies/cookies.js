// Cookie management
let cookiesInfo = document.getElementById('cookies_info');
let cookiesButton = document.getElementById('cookies_info_button');
let cookiesAcceptButton = document.getElementById('accept');
let cookiesDeclineButton = document.getElementById('decline');

const cookieStorage = {
    getItem: (key) => {
        const cookies = document.cookie
            .split(';')
            .map(cookie => cookie.split('='))
            .reduce((acc, [key, value]) => ({ ...acc, [key.trim()]: value}), {});

            return cookies[key]
    },
    setItem: (key, value) => {
        document.cookie = `${key}=${value}; expires=${Date()}; path=/;`;
    },
    deleteItem: (key, value=true) => {
        document.cookie = `${key}=${value}; expires=Thu, 18 Dec 2013 12:00:00 UTC; path=/;`
    }
}

const localStorageType = localStorage;
const cookiesConsentGiven = 'cookies_consent'

const storageType = cookieStorage;
const consentPropertyName = 'jdc_consent'

const shouldShowPopup = () => !localStorageType.getItem(cookiesConsentGiven);
const saveToStorage = () => storageType.setItem(consentPropertyName, true);
const saveToLocalStorageAgreed = () => localStorageType.setItem(cookiesConsentGiven, true)
const saveToLocalStorageDisagreed = () => localStorageType.setItem(cookiesConsentGiven, false)

function cookiesInfoDisAppear() {
    cookiesInfo.style.transform = "translateY(450px)";
};

function cookiesInfoAppear() {
    cookiesInfo.style.transform = "translateY(0px)";
};

window.onload = () => {
    if (shouldShowPopup(localStorageType)) {
        setTimeout(() => {
            cookiesInfoAppear();
        }, 1000);
    };

    const acceptFn = event => {
        saveToStorage(storageType);
        saveToLocalStorageAgreed(localStorageType);
        cookiesInfoDisAppear();
    };
    
    const deleteAllCookies = () => {
        document.cookie.split(";").forEach(function(c) { document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/; domain=.000webhostapp.com"); });
        document.cookie.split(";").forEach(function(c) { document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); });
        saveToLocalStorageDisagreed();
        cookiesInfoDisAppear();
    };
    
    cookiesAcceptButton.addEventListener('click', acceptFn);
    cookiesDeclineButton.addEventListener('click', deleteAllCookies);
};

cookiesButton.onclick = function() {cookiesInfoAppear()};