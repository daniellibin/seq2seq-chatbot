/*聊天信息*/
function show(headSrc, str, className) {
	var html = "<div class=" + className + "><div class='msg'><img src=" + headSrc + " />" +
		"<p><i class='msg_input'></i>" + str + "</p></div></div>";
	upView(html);
}
/*更新视图*/
function upView(html) {
	$('.message').append(html);
	$('body,html').animate({
		scrollTop: $('.message').outerHeight() - window.innerHeight
	}, 200); // 自动将页面移动到最底部
	// $("body").animate({scrollTop:top})不被Firefox支持,而被chrome支持
	// $("html").animate({scrollTop:top})而被chrome支持,而被Firefox支持
}

var flag = true; // 防止连续点击提交消息
var message = ''; // 接收接口返回的数据


$(function () {
	$('#inputVal').focus();
	$('.footer').on('keyup', 'input', function () {
		if ($(this).val().length > 0) {
			$(this).next().css('background', '#114F8E');

		} else {
			$(this).next().css('background', '#ddd');
		}
	});

	$('.footer p').click(getMessage);
	$(document).keyup(function (ev) {
		if (ev.keyCode == 13) {
			getMessage();
		}
	})

	function getMessage() {
		var val = $('#inputVal').val();
		if (val == '')
			return;
		if (flag) {
			flag = false;

			show("./chatImages/woman.png", $(".footer input").val(), "show");
			// 替换为各自的接口地址
			var url = "http://127.0.0.1:8000/api/chatbot";
			
			$(".footer input").val("").next().css('background', '#ddd'); //清空input
			$.ajax({
				type: "get",
				dataType: "json",
				async: true,
				url: url,
				data: {
					infos: val,
				},
				complete: function (data) {
					flag = true;
					message = data.responseText
						setTimeout(function () {
							show("chatImages/man.png", message, "send");
						}, 500);
				}
			});
		}
	}
});
