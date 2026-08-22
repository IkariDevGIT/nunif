function on_recap_checked(e) {
    enable_buttons();
}
function on_turnstile_success(token) {
    $("#turnstile_response").val(token);
    enable_buttons();
}

function disable_buttons() {
    $("#submit-button").prop("disabled", true);
}
function enable_buttons() {
    $("#submit-button").prop("disabled", false);
}


$(function (){
    var g_expires = 365;
    var recaptcha_js = "https://www.recaptcha.net/recaptcha/api.js";
    var turnstile_widgetid;
    var turnstile_is_enabled = false;
    var g_last_fingerprint = "";
    var g_error_count = 0;

    function get_fingerprint() {
        var obj = {
            style: $("input[name=style]:checked").val(),
            noise: $("input[name=noise]:checked").val(),
            scale: $("input[name=scale]:checked").val(),
            format: $("input[name=format]:checked").val(),
            url: $("#url").val()
        };
        var file = $("#file").get(0).files[0];
        if (file) {
            obj.file = {name: file.name, size: file.size, lastModified: file.lastModified};
        }
        return JSON.stringify(obj);
    }

    function parse_error_html(html) {
        var title_match = /<title\b[^>]*>([\s\S]*?)<\/title>/i.exec(html);
        var pre_match = /<body\b[^>]*>[\s\S]*?<pre\b[^>]*>([\s\S]*?)<\/pre>/i.exec(html);
        var title = title_match ? title_match[1].trim() : "";
        var body_pre = pre_match ? pre_match[1].trim() : "";
        var text = "";
        if (title) text += title;
        if (body_pre) text += (text ? "\n" : "") + body_pre;
        return text;
    }

    function set_error_message(text, second, is_html, status) {
        var count = ++g_error_count;
        if (!second) second = 2;
        var display_text = is_html ? parse_error_html(text) : text;
        
        if (display_text) {
            $("#error_message").text(display_text).show();
        } else if (status) {
            $("#error_message").text("HTTP Error (" + status + ")").show();
        } else {
            $("#error_message").text(text).show();
        }

        setTimeout(function() {
            if (count == g_error_count) {
                $("#error_message").fadeOut();
            }
        }, second * 1000);
    }

    function reset_turnstile() {
        turnstile.reset(turnstile_widgetid);
    }
    function clear_file() {
        $("#file").val("");
    }
    function clear_url() {
        $("#url").val("");
    }
    function on_change_style(e) {
        var checked = $("input[name=style]:checked");
        if (checked.val() != "photo") {
            $(".main-title").text("waifu2x");
        } else {
            $(".main-title").html("w<s>/a/</s>ifu2x");
        }
        $.cookie("style", checked.val(), {expires: g_expires});
    }
    function on_change_noise_level(e)
    {
        var checked = $("input[name=noise]:checked");
        $.cookie("noise", checked.val(), {expires: g_expires});
    }
    function on_change_scale_factor(e)
    {
        var checked = $("input[name=scale]:checked");
        $.cookie("scale", checked.val(), {expires: g_expires});
    }
    function on_change_format(e)
    {
        var checked = $("input[name=format]:checked");
        $.cookie("format", checked.val(), {expires: g_expires});
    }
    function commit_recap_response()
    {
        if (typeof grecaptcha != "undefined") {
            console.log("recaptcha: enabled")
            $("#recap_response").val(grecaptcha.getResponse());
            grecaptcha.reset();
            disable_buttons();
        } else {
            console.log("recaptcha: disabled")
        }
    }
    function commit_turnstile_response()
    {
        if (turnstile_is_enabled) {
            console.log("turnstile: enabled")
            reset_turnstile();
            disable_buttons();
        } else {
            console.log("turnstile: disabled")
        }
    }
    function restore_from_cookie()
    {
        if ($.cookie("style")) {
            $("input[name=style]").filter("[value=" + $.cookie("style") + "]").prop("checked", true);
        }
        if ($.cookie("noise")) {
            $("input[name=noise]").filter("[value=" + $.cookie("noise") + "]").prop("checked", true);
        }
        if ($.cookie("scale")) {
            $("input[name=scale]").filter("[value=" + $.cookie("scale") + "]").prop("checked", true);
        }
        if ($.cookie("format")) {
            $("input[name=format]").filter("[value=" + $.cookie("format") + "]").prop("checked", true);
        }
    }
    function uuid() 
    {
        // ref: http://stackoverflow.com/a/2117523
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
            return v.toString(16);
        });
    }
    function extract_filename(disposition, default_val)
    {
        if (disposition && disposition.indexOf('filename') != -1) {
            var reg = /filename[^;=\n]*=(?:(\\?['"])(.*?)\1|(?:[^\s]+'.*?')?([^;\n]*))/i;
            var matches = reg.exec(disposition);
            if (matches && matches.length >= 4 && matches[3]) {
                return decodeURI(matches[3]);
            }
        }
        return default_val;
    }
    function webp_to_png(blob, callback) {
        console.log("Use webp_to_png")
        var url = URL.createObjectURL(blob);
        var img = new Image();
        img.onload = function() {
            var width = img.width;
            var height = img.height;
            var canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            var gl = canvas.getContext("webgl");
            if (gl) {
                gl.activeTexture(gl.TEXTURE0);
                var texture = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                var framebuffer = gl.createFramebuffer();
                gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
                var image_data = new ImageData(width, height);
                gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, image_data.data);
                
                var has_alpha = false;
                for (var i = 3; i < image_data.data.length; i += 4) {
                    if (image_data.data[i] < 255) {
                        has_alpha = true;
                        break;
                    }
                }

                var out_canvas = document.createElement("canvas");
                out_canvas.width = width;
                out_canvas.height = height;
                var out_ctx = out_canvas.getContext("2d", {alpha: has_alpha});
                out_ctx.putImageData(image_data, 0, 0);
                canvas = out_canvas;

                gl.deleteTexture(texture);
                gl.deleteFramebuffer(framebuffer);
            } else {
                canvas.getContext("2d").drawImage(img, 0, 0);
            }
            canvas.toBlob(function(pngBlob) {
                URL.revokeObjectURL(url);
                callback(pngBlob);
            }, "image/png");
        };
        img.onerror = function() {
            URL.revokeObjectURL(url);
            callback(blob);
        };
        img.src = url;
    }
    function request_image() 
    {
        if (typeof window.URL.createObjectURL == "undefined" ||
            typeof window.Blob == "undefined" ||
            typeof window.XMLHttpRequest == "undefined" ||
            typeof window.URL.revokeObjectURL == "undefined")
        {
            return;
        }

        var current_fingerprint = get_fingerprint();
        if (current_fingerprint == g_last_fingerprint) {
            set_error_message("Same request.");
            return;
        }

        var file = $("#file").get(0).files[0];
        var url = $("#url").val();
        if (!file && !url) {
            set_error_message("Select a file or enter a URL.");
            return;
        }

        $("#error_message").hide();
        var $item = $("<div>").addClass("result-item");
        var $loading = $("<div>").addClass("item-loading").append($("<img>").attr("src", "loading.webp"));
        $item.append($loading);
        $("#result").prepend($item);
        $("#result_wrapper").show();

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/api', true);
        xhr.responseType = 'arraybuffer';
        xhr.onload = function(e) {
            $loading.remove();
            if (!turnstile_is_enabled && typeof grecaptcha == "undefined") {
                enable_buttons();
            }
            if (this.status == 200) {
                g_last_fingerprint = current_fingerprint;
                var contentType = this.getResponseHeader("Content-Type");
                var blob = new Blob([this.response], {type : contentType});
                var format = $("input[name=format]:checked").val();
                var disposition = this.getResponseHeader("Content-Disposition");

                var proceed = function(finalBlob) {
                    var finalType = finalBlob.type;
                    var ext = (finalType == "image/webp") ? "webp" : "png";
                    var filename = extract_filename(disposition, uuid() + "." + ext);
                    if (ext == "png") {
                        filename = filename.replace(/\.webp$/i, ".png");
                        if (!filename.match(/\.png$/i)) filename += ".png";
                    } else {
                        filename = filename.replace(/\.png$/i, ".webp");
                        if (!filename.match(/\.webp$/i)) filename += ".webp";
                    }

                    var url = URL.createObjectURL(finalBlob);
                    var img = $("<img>").attr("src", url).attr("alt", filename).attr("title", filename).click(function() {
                        $(this).toggleClass("zoomed");
                    });
                    var link = $("<a>")
                        .attr("href", url)
                        .attr("download", filename)
                        .addClass("download-link")
                        .text("Download " + filename);
                    $item.append(link).append(img);
                };

                if (format == "0" && contentType == "image/webp") {
                    webp_to_png(blob, proceed);
                } else {
                    proceed(blob);
                }
            } else {
                var contentType = this.getResponseHeader("Content-Type");
                var error_text = "HTTP Error (" + this.status + ")";
                var is_html = (contentType && contentType.indexOf("text/html") !== -1);
                try {
                    var decoded = new TextDecoder().decode(new Uint8Array(this.response));
                    if (decoded) {
                        error_text = is_html ? parse_error_html(decoded) : decoded;
                        if (!error_text) error_text = "HTTP Error (" + this.status + ")";
                    }
                } catch (err) {}
                var $error = $("<div>").addClass("item-error").text(error_text);
                $item.append($error);
            }
        };
        xhr.onerror = function() {
            $loading.remove();
            if (!turnstile_is_enabled && typeof grecaptcha == "undefined") {
                enable_buttons();
            }
            var $error = $("<div>").addClass("item-error").text("Network Error");
            $item.append($error);
        };
        commit_recap_response();
        commit_turnstile_response();
        xhr.send(new FormData($("form").get(0)));
    }
    function load_recaptcha()
    {
        $.ajax({
            url: "/recaptcha_state.json",
            type: "GET",
            dataType: "json",
        }).done(function (data) {
            if (data.recaptcha_enabled) {
                // setup recaptcha
                console.log("recaptcha is enabled");
                $("<div>").attr({
                    "class": "g-recaptcha",
                    "data-sitekey": data.recaptcha_site_key,
                    "data-callback": "on_recap_checked"
                }).appendTo("#recap_container");
                $("<script>").attr({
                    type: "text/javascript",
                    src: recaptcha_js
                }).appendTo(document.head);
                disable_buttons();
            } else if (data.turnstile_enabled) {
                // setup turnstile
                console.log("turnstile is enabled");
                turnstile_is_enabled = true;
                turnstile_widgetid = turnstile.render('#turnstile_container', {
                     sitekey: data.turnstile_site_key,
                     callback: on_turnstile_success,
                });
                disable_buttons();
            } else {
                console.log("recaptcha/turnstile is disabled");
            }
        }).fail(function (e) {
            console.log(e)
        });
    }
    function set_param()
    {
        var uri = URI(window.location.href);
        var url = uri.query(true)["url"];
        var style = uri.query(true)["style"];
        var noise = uri.query(true)["noise"];
        var scale = uri.query(true)["scale"];
        var format = uri.query(true)["format"];
        if (url) {
            $("input[name=url]").val(url);
        }
        if (style) {
            $("input[name=style]").filter("[value=" + style + "]").prop("checked", true);
        }
        if (noise) {
            $("input[name=noise]").filter("[value=" + noise + "]").prop("checked", true);
        }
        if (scale) {
            $("input[name=scale]").filter("[value=" + scale + "]").prop("checked", true);
        }
        if (format) {
            $("input[name=format]").filter("[value=" + format + "]").prop("checked", true);
        }
    }
    $("#url").change(clear_file);
    $("#file").change(clear_url);
    $("input[name=style]").change(on_change_style);
    $("input[name=noise]").change(on_change_noise_level);
    $("input[name=scale]").change(on_change_scale_factor);
    $("input[name=format]").change(on_change_format);
    $("form").submit(function(e) {
        if ($("#result").length > 0) {
            e.preventDefault();
            request_image();
        } else {
            commit_recap_response();
            commit_turnstile_response();
            // allow default submit
        }
    });
    $("#clear_results").click(function(e) {
        e.preventDefault();
        e.stopPropagation();
        location.reload();
    });
    $(document).on({
        dragover: function() { return false; },
        drop: function(e) {
            if (!(e.originalEvent.dataTransfer && e.originalEvent.dataTransfer.files.length)) {
                return false;
            }
            var file = e.originalEvent.dataTransfer;
            if (file.files.length > 0 && file.files[0].type.match(/image/)) {
                var files = new DataTransfer();
                files.items.add(file.files[0]);
                $("#file").get(0).files = files.files;
                $("#file").trigger("change");
                return false;
            } else {
                return false;
            }
        }
    });

    restore_from_cookie();
    on_change_style();
    on_change_scale_factor();
    on_change_noise_level();
    on_change_format();
    set_param();
    load_recaptcha();
})
