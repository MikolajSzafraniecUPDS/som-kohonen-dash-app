window.addEventListener(
    'beforeunload', () => {
        document.querySelector("#clear_cache_btn").click();
    }
);

//window.addEventListener(
//    'beforeunload', (event) => {
//        document.querySelector("#clear_cache_btn").click();
//        event.returnValue = "returnString";
//        return "returnString";
//    }
//);