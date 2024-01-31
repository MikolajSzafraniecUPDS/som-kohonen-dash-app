window.addEventListener(
    'beforeunload', () => {
        document.querySelector("#clear_cache_btn").click()
    }
);