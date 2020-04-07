  (function (doc, win) {
        var docEl = doc.documentElement, // html
            resizeEvt = 'orientationchange' in window ? 'orientationchange' : 'resize', // orientationchange:用户水平或者垂直翻转设备（即方向发生变化）时触发的事件。
            recalc = function () {
                var clientWidth = docEl.clientWidth; // 拿到浏览器的css像素值
                if (!clientWidth) return;
                if(clientWidth>=720){
                    docEl.style.fontSize = '100px';
                }else{
                    docEl.style.fontSize = 100 * (clientWidth / 720) + 'px';
                }
            };

        if (!doc.addEventListener) return;
        win.addEventListener(resizeEvt, recalc, false);
        doc.addEventListener('DOMContentLoaded', recalc, false);
    })(document, window);

