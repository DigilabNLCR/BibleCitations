window.onscroll = function() {menuScrollFunction()};

function menuScrollFunction() {
  if (window.innerWidth < 930) {
    document.getElementById("sticky-menu-block").style.display = "none";
  } else if (document.body.scrollTop > 656 || document.documentElement.scrollTop > 656) {
    document.getElementById("sticky-navbar").style.height = "64px";
    document.getElementById("sticky-navbar-container").style.marginTop = "0px";
    document.getElementById("sticky-menu-block").style.display = "block";
    document.getElementById("sticky-menu-block").style.position = "fixed";
    document.getElementById("sticky-menu-block").style.top = "0px";
    document.getElementById("sticky-menu-block").style.opacity = "1";
    document.getElementById("sticky-menu-items").style.top = "31px";
  } else {
    document.getElementById("sticky-menu-block").style.display = "none";
  }
}